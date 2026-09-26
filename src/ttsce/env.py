"""Oracle-hiding replay environment over the frozen response table.

Scope (fixed in docs/protocol.md):
  * fixed candidate pool: the released candidates of each problem;
  * history-independent queries: every policy executes a schedule of queries
    that is fixed *before* any response is seen (a deterministic function of
    the schedule spec, the problem id and the order seed). A policy may only
    decide to execute the next scheduled query or to stop.

Because the query sequence cannot depend on observed responses, replaying
pre-collected i.i.d. responses is exactly equivalent to querying the models
online. Policies whose next query depends on history are out of scope for this
runner (they would need a live online runner).

The policy only ever receives an `Observation`: answers of generated
candidates (in generation order) and verdicts of executed verifier calls.
Gold labels, candidate ids and unexecuted cells are not reachable from it.
"""
from dataclasses import dataclass, field

import numpy as np

from .table import ABSTAIN, NO_VERDICT, ProblemArrays


# ---------------------------------------------------------------- schedules
def schedule_generate_only(n_max):
    """G: generate candidates one at a time (self-consistency style)."""
    return [("gen",)] * n_max


def schedule_interleaved(n_max, k):
    """GVk: generate a candidate, then verify it k times, then the next one."""
    out = []
    for c in range(n_max):
        out.append(("gen",))
        out.extend([("ver", c)] * k)
    return out


def schedule_fixed(n, v):
    """Fixed allocation: n generations, then v verifier calls per candidate (round robin)."""
    return [("gen",)] * n + [("ver", c) for _ in range(v) for c in range(n)]


# ------------------------------------------------------------- observation
@dataclass
class Observation:
    answers: list = field(default_factory=list)       # int letter or ABSTAIN, generation order
    verdicts: list = field(default_factory=list)      # per generated candidate: list of 0/1/NO_VERDICT
    gen_in_tok: list = field(default_factory=list)    # prompt length of each generation call
    gen_out_tok: list = field(default_factory=list)   # completion length of each generated candidate
    ver_in_tok: list = field(default_factory=list)    # per generated candidate: verifier prompt length (0 if never verified)
    ver_out_tok: list = field(default_factory=list)   # per generated candidate: completion length of each verifier call

    def n_valid_votes(self):
        return sum(a != ABSTAIN for a in self.answers)


@dataclass
class Cost:
    n_gen: int = 0
    n_ver: int = 0
    gen_in: int = 0
    gen_out: int = 0
    ver_in: int = 0
    ver_out: int = 0

    @property
    def total_tokens(self):
        return self.gen_in + self.gen_out + self.ver_in + self.ver_out

    @property
    def output_tokens(self):
        return self.gen_out + self.ver_out


class ScheduleExhausted(Exception):
    pass


class ReplayEnv:
    def __init__(self, arrays: ProblemArrays, order_seed: int, schedule):
        self.__a = arrays
        rng = np.random.default_rng([int(order_seed), int(arrays.problem)])
        n_cand, n_ver = len(arrays.answers), arrays.n_verified
        # verified pool first, then the generation-only remainder; both shuffled
        self.__cand_order = np.concatenate([rng.permutation(n_ver), n_ver + rng.permutation(n_cand - n_ver)])
        self.__rep_order = [rng.permutation(arrays.verdicts.shape[1]) for _ in range(n_ver)]
        self.__next_rep = np.zeros(n_ver, dtype=int)
        self.__schedule = list(schedule)
        self.__t = 0
        self.__seen_gen_prompt = False
        self.__seen_ver_prompt = set()
        self.cost = Cost()
        self.cached_input_tokens = 0   # input tokens if identical prompts are prefix-cached
        self.obs = Observation()
        self.__audit = []              # revealed cells, for leakage audits only

    def audit_log(self):
        """Cells revealed so far as ("gen", j) / ("ver", j, r). Audit/tests only."""
        return list(self.__audit)

    @property
    def steps_done(self):
        return self.__t

    @property
    def steps_left(self):
        return len(self.__schedule) - self.__t

    def step(self):
        if self.__t >= len(self.__schedule):
            raise ScheduleExhausted
        act = self.__schedule[self.__t]
        self.__t += 1
        a = self.__a
        if act[0] == "gen":
            k = len(self.obs.answers)
            j = self.__cand_order[k]
            self.__audit.append(("gen", int(j)))
            self.obs.answers.append(int(a.answers[j]))
            self.obs.verdicts.append([])
            self.obs.gen_in_tok.append(int(a.gen_in[j]))
            self.obs.gen_out_tok.append(int(a.gen_out[j]))
            self.obs.ver_in_tok.append(0)
            self.obs.ver_out_tok.append([])
            self.cost.n_gen += 1
            self.cost.gen_in += int(a.gen_in[j])
            self.cost.gen_out += int(a.gen_out[j])
            if not self.__seen_gen_prompt:
                self.cached_input_tokens += int(a.gen_in[j])
                self.__seen_gen_prompt = True
        else:
            k = act[1]
            if k >= len(self.obs.answers):
                raise ValueError("schedule verifies a candidate that was not generated yet")
            j = self.__cand_order[k]
            if j >= a.n_verified:
                raise ValueError("candidate outside the verified pool")
            r_idx = self.__next_rep[j]
            if r_idx >= len(self.__rep_order[j]):
                raise ValueError("verifier repeats exhausted for this candidate")
            r = self.__rep_order[j][r_idx]
            self.__next_rep[j] += 1
            self.__audit.append(("ver", int(j), int(r)))
            self.obs.verdicts[k].append(int(a.verdicts[j, r]))
            self.obs.ver_in_tok[k] = int(a.ver_in[j])
            self.obs.ver_out_tok[k].append(int(a.ver_out[j, r]))
            self.cost.n_ver += 1
            self.cost.ver_in += int(a.ver_in[j])
            self.cost.ver_out += int(a.ver_out[j, r])
            if j not in self.__seen_ver_prompt:
                self.cached_input_tokens += int(a.ver_in[j])
                self.__seen_ver_prompt.add(j)
        return act

    def run_all(self):
        while self.steps_left:
            self.step()
        return self.obs


__all__ = ["ReplayEnv", "Observation", "Cost", "ScheduleExhausted", "schedule_generate_only",
           "schedule_interleaved", "schedule_fixed", "ABSTAIN", "NO_VERDICT"]
