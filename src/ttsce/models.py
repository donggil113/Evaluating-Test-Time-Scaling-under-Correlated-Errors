"""Aggregators and answer-posterior models.

Everything here maps an Observation (revealed data only) to a decision.
Parameters of the Bayesian scorers are fit on dev problems by `fit_*`
functions in scripts/, which are the only place gold labels are used.

Answer posterior (4-way multiple choice, one correct letter y):
  vote term   : candidate accuracy q ~ Beta(aq, bq) per problem, wrong answers
                uniform over the 3 other letters  ->  Beta-binomial in n_y
  verdict term: per verified candidate with `yes` accepts and `no` rejects,
                IND  : i.i.d. Bernoulli(t) if correct, Bernoulli(f) if wrong
                CORR : Beta-binomial (exchangeable, per-candidate propensity
                       alpha ~ Beta) as in the partially-correlated cascade model
  calibrated score s_a = w_vote * vote_a + w_ver * ver_a, p = softmax(s).
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import betaln, betainc

from .table import ABSTAIN, NO_VERDICT

LOG3 = np.log(3.0)


# ------------------------------------------------------------ helpers
def _first_seen_rank(answers):
    rank = {}
    for i, a in enumerate(answers):
        if a != ABSTAIN and a not in rank:
            rank[a] = i
    return rank


def _yes_no(vs):
    y = sum(1 for v in vs if v == 1)
    n = sum(1 for v in vs if v == 0)
    return y, n


def _argmax_tiebreak(score, answers):
    """argmax over letters present in `answers`; ties -> plurality, then first seen."""
    rank = _first_seen_rank(answers)
    if not rank:
        return ABSTAIN
    votes = np.bincount([a for a in answers if a != ABSTAIN], minlength=4)
    return max(rank, key=lambda a: (score[a], votes[a], -rank[a]))


# ---------------------------------------------------------- aggregators
def agg_majority(answers, verdicts=None):
    return _argmax_tiebreak(np.zeros(4), answers)


def agg_best_of_n(answers, verdicts):
    """Highest mean acceptance among verified candidates; unverified -> MV fallback."""
    best, best_key = None, None
    votes = np.bincount([a for a in answers if a != ABSTAIN], minlength=4)
    rank = _first_seen_rank(answers)
    for a, vs in zip(answers, verdicts):
        if a == ABSTAIN:
            continue
        y, n = _yes_no(vs)
        if y + n == 0:
            continue
        key = (y / (y + n), votes[a], -rank[a])
        if best_key is None or key > best_key:
            best, best_key = a, key
    return agg_majority(answers) if best is None else best


def agg_weighted_majority(answers, verdicts):
    """Sum of per-candidate acceptance rates per answer (unverified candidates add 0)."""
    score = np.zeros(4)
    for a, vs in zip(answers, verdicts):
        if a == ABSTAIN:
            continue
        y, n = _yes_no(vs)
        if y + n:
            score[a] += y / (y + n)
    return _argmax_tiebreak(score, answers)


def adaptive_consistency_prob(answers):
    """Adaptive-Consistency (Aggarwal et al., 2023) Beta criterion:
    P(p_top1 > p_top2) with Beta(v1+1, v2+1) over the two leading answers."""
    votes = np.sort(np.bincount([a for a in answers if a != ABSTAIN], minlength=4))[::-1]
    v1, v2 = votes[0], votes[1]
    if v1 == 0:
        return 0.0
    return float(1.0 - betainc(v1 + 1, v2 + 1, 0.5))


# ------------------------------------------------------- Bayesian scorer
@dataclass
class BayesParams:
    kind: str            # "IND" or "CORR"
    aq: float
    bq: float
    t: float = 0.5       # IND: accept prob. of a correct candidate
    f: float = 0.5       # IND: accept prob. of a wrong candidate
    a1: float = 1.0      # CORR: Beta params of correct-candidate acceptance propensity
    b1: float = 1.0
    a0: float = 1.0      # CORR: Beta params of wrong-candidate acceptance propensity
    b0: float = 1.0
    w_vote: float = 1.0
    w_ver: float = 1.0

    def llr(self, y, n):
        """log P(verdicts | candidate correct) - log P(verdicts | wrong)."""
        if y + n == 0:
            return 0.0
        if self.kind == "IND":
            return y * np.log(self.t / self.f) + n * np.log((1 - self.t) / (1 - self.f))
        return (betaln(self.a1 + y, self.b1 + n) - betaln(self.a1, self.b1)
                - betaln(self.a0 + y, self.b0 + n) + betaln(self.a0, self.b0))

    def vote_term(self, counts):
        n = counts.sum()
        return betaln(self.aq + counts, self.bq + n - counts) - (n - counts) * LOG3


class IncrementalScorer:
    """Posterior over the 4 letters, updated from the revealed stream only."""

    def __init__(self, params: BayesParams):
        self.p = params
        self.counts = np.zeros(4)
        self.ver = np.zeros(4)
        self._yn = []          # per generated candidate [yes, no]
        self._ans = []
        self._llr = []

    def update(self, obs, act):
        if act[0] == "gen":
            a = obs.answers[-1]
            self._ans.append(a)
            self._yn.append([0, 0])
            self._llr.append(0.0)
            if a != ABSTAIN:
                self.counts[a] += 1
        else:
            k = act[1]
            v = obs.verdicts[k][-1]
            if v == NO_VERDICT:
                return
            self._yn[k][0 if v == 1 else 1] += 1
            a = self._ans[k]
            if a == ABSTAIN:
                return
            new = self.p.llr(*self._yn[k])
            self.ver[a] += new - self._llr[k]
            self._llr[k] = new

    def logits(self):
        return self.p.w_vote * self.p.vote_term(self.counts) + self.p.w_ver * self.ver

    def probs(self):
        s = self.logits()
        s = s - s.max()
        e = np.exp(s)
        return e / e.sum()


def bayes_terms(params, answers, verdicts):
    """(vote_term, verdict_term) vectors for a full observation (batch form)."""
    counts = np.bincount([a for a in answers if a != ABSTAIN], minlength=4).astype(float)
    ver = np.zeros(4)
    for a, vs in zip(answers, verdicts):
        if a != ABSTAIN:
            ver[a] += params.llr(*_yes_no(vs))
    return params.vote_term(counts), ver


def agg_bayes(params):
    def f(answers, verdicts):
        v, r = bayes_terms(params, answers, verdicts)
        return _argmax_tiebreak(params.w_vote * v + params.w_ver * r, answers)
    return f


AGGREGATORS = {"MV": agg_majority, "BoN": agg_best_of_n, "WMV": agg_weighted_majority}
