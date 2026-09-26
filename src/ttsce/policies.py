"""Policies executed against ReplayEnv. Every policy follows a fixed schedule
and may only choose when to stop; decisions use the Observation only."""
import time

import numpy as np

from .env import ReplayEnv, schedule_fixed, schedule_generate_only, schedule_interleaved
from .models import IncrementalScorer, adaptive_consistency_prob, agg_majority, _argmax_tiebreak


def make_schedule(spec):
    """spec: ("G", n_max) | ("GV", n_max, k) | ("FIX", n, v)."""
    if spec[0] == "G":
        return schedule_generate_only(spec[1])
    if spec[0] == "GV":
        return schedule_interleaved(spec[1], spec[2])
    if spec[0] == "FIX":
        return schedule_fixed(spec[1], spec[2])
    raise ValueError(spec)


def _result(env, answer, cpu):
    c = env.cost
    return {"answer": int(answer), "n_gen": c.n_gen, "n_ver": c.n_ver, "total_tokens": c.total_tokens,
            "output_tokens": c.output_tokens, "cached_tokens": env.cached_input_tokens + c.output_tokens,
            "gen_tokens": c.gen_in + c.gen_out, "ver_tokens": c.ver_in + c.ver_out, "controller_cpu_s": cpu}


def run_fixed(arrays, seed, n, v, agg):
    env = ReplayEnv(arrays, seed, make_schedule(("FIX", n, v)))
    obs = env.run_all()
    t0 = time.process_time()
    ans = agg(obs.answers, obs.verdicts)
    return _result(env, ans, time.process_time() - t0)


def run_stopping(arrays, seed, spec, params, tau):
    """Calibrated stopping: stop once max posterior >= tau (checked after every query)."""
    env = ReplayEnv(arrays, seed, make_schedule(spec))
    scorer = IncrementalScorer(params)
    cpu = 0.0
    while env.steps_left:
        act = env.step()
        t0 = time.process_time()
        scorer.update(env.obs, act)
        stop = scorer.probs().max() >= tau
        cpu += time.process_time() - t0
        if stop:
            break
    t0 = time.process_time()
    ans = _argmax_tiebreak(scorer.logits(), env.obs.answers)
    return _result(env, ans, cpu + time.process_time() - t0)


def run_adaptive_consistency(arrays, seed, n_max, threshold):
    env = ReplayEnv(arrays, seed, make_schedule(("G", n_max)))
    cpu = 0.0
    while env.steps_left:
        env.step()
        t0 = time.process_time()
        stop = adaptive_consistency_prob(env.obs.answers) >= threshold
        cpu += time.process_time() - t0
        if stop:
            break
    t0 = time.process_time()
    ans = agg_majority(env.obs.answers)
    return _result(env, ans, cpu + time.process_time() - t0)


# ---------------------------------------------------------------- dev tools
def stopping_trajectory(arrays, seed, spec, params=None, ac=False):
    """Run the whole schedule once and record, after every step, the stopping
    statistic, the answer that would be returned and the cost so far.
    Used only for dev-side sweeps over tau; equivalence with the online
    policies is asserted in tests/test_replay.py."""
    env = ReplayEnv(arrays, seed, make_schedule(spec))
    scorer = None if ac else IncrementalScorer(params)
    stat, ans, tot, out, cached = [], [], [], [], []
    while env.steps_left:
        act = env.step()
        if ac:
            stat.append(adaptive_consistency_prob(env.obs.answers))
            ans.append(agg_majority(env.obs.answers))
        else:
            scorer.update(env.obs, act)
            stat.append(scorer.probs().max())
            ans.append(_argmax_tiebreak(scorer.logits(), env.obs.answers))
        tot.append(env.cost.total_tokens)
        out.append(env.cost.output_tokens)
        cached.append(env.cached_input_tokens + env.cost.output_tokens)
    return {k: np.asarray(v) for k, v in
            dict(stat=stat, answer=ans, total_tokens=tot, output_tokens=out, cached_tokens=cached).items()}


def stop_index(stat, tau):
    hit = np.nonzero(stat >= tau)[0]
    return int(hit[0]) if len(hit) else len(stat) - 1


def prefix_eval(obs, n, v, agg):
    """Fixed (n, v) allocation evaluated on a prefix of a full fixed-schedule
    observation (identical reveal set; asserted in tests)."""
    answers = obs.answers[:n]
    verdicts = [vs[:v] for vs in obs.verdicts[:n]]
    ans = agg(answers, verdicts)
    gen_t = sum(obs.gen_in_tok[:n]) + sum(obs.gen_out_tok[:n])
    ver_out = sum(sum(x[:v]) for x in obs.ver_out_tok[:n])
    ver_in = sum(obs.ver_in_tok[k] * min(v, len(obs.ver_out_tok[k])) for k in range(min(n, len(obs.ver_in_tok))))
    n_ver = sum(min(v, len(x)) for x in obs.ver_out_tok[:n])
    cached = (obs.gen_in_tok[0] if n else 0) + sum(obs.gen_out_tok[:n]) + ver_out + \
        (sum(obs.ver_in_tok[k] for k in range(min(n, len(obs.ver_in_tok)))) if v else 0)
    return {"answer": int(ans), "n_gen": n, "n_ver": n_ver, "total_tokens": gen_t + ver_in + ver_out,
            "output_tokens": sum(obs.gen_out_tok[:n]) + ver_out, "cached_tokens": cached,
            "gen_tokens": gen_t, "ver_tokens": ver_in + ver_out}
