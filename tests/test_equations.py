"""M3 acceptance for the constants domain: PSLQ relations (plan 7, 12)."""
import random

import mpmath as mp

from rde import constants as C
from rde.equations import pslq_relation


def test_zeta2_six_times_equals_pi_squared():
    names = ["zeta2", "pi^2"]
    rel = pslq_relation(lambda: [mp.zeta(2), mp.pi ** 2], names, target="pi^2")
    assert rel == (6, -1) or rel == (-6, 1)


def _cf_at_current_precision(p, q):
    """pcf value at the current mpmath precision, doubling depth until it converges."""
    depth = 100
    while True:
        K = C.pcf_converged(p, q, depth, dps=mp.mp.dps)
        if K is not None:
            return K
        depth *= 2
        assert depth <= 6400, "continued fraction did not converge"


def test_four_over_pi_continued_fraction_gives_pi_times_k_equals_four():
    # a_n = 2n+1 (p(n) = 1 + 2n), b_n = n^2 (q(n) = n^2): K = 4/pi
    p, q = (1, 2), (0, 0, 1)
    rel = pslq_relation(lambda: [mp.mpf(1), mp.pi, _cf_at_current_precision(p, q),
                                 mp.pi * _cf_at_current_precision(p, q)],
                        ["1", "pi", "K", "pi*K"], target="pi*K")
    assert rel is not None
    c0, c1, c2, c3 = rel
    assert (c1, c2) == (0, 0) and c0 == -4 * c3


def test_four_random_hundred_digit_numbers_give_no_relation():
    rnd = random.Random(12345)
    with mp.workdps(120):
        vals = [mp.mpf(rnd.randrange(10 ** 99, 10 ** 100)) / 10 ** 100 for _ in range(4)]
        assert pslq_relation(lambda: vals, ["x", "y", "z", "w"], target="w") is None
