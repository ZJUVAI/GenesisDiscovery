"""M3 acceptance for the constants domain: continued fractions and PSLQ relations."""
import mpmath as mp

from rde import constants as C


def test_four_over_pi_continued_fraction():
    ok, detail = C.check_known_4_over_pi()
    assert ok, detail


def test_random_hundred_digit_numbers_give_no_relation():
    with mp.workdps(120):
        x = mp.mpf(mp.sqrt(2) + mp.mpf("0.3141592653589793"))
        K = mp.mpf(mp.log(3) + mp.mpf("0.1234567891234567") * mp.sqrt(5))
        assert C.relation_search(K, x, dps=60, maxcoeff=500) is None


def test_zeta2_relation_with_pi_squared():
    with mp.workdps(80):
        vals = [mp.zeta(2) * 6, mp.pi ** 2]
        rel = mp.pslq(vals, maxcoeff=1000, maxsteps=10000)
        assert rel == [1, -1]


def test_mitm_screen_finds_constructed_relation():
    from rde.constants_mitm import Index, verify
    with mp.workdps(80):
        x = mp.pi
        K = -(1 + 2 * x) / (3 + x)  # relation (c0, c1, c2, c3) = (1, 2, 3, 1)
        idx = Index(float(x), bound=12)
        cands = idx.lookup(float(K))
        assert (1, 2, 3, 1) in cands
        assert verify((1, 2, 3, 1), x, K)
        assert not verify((1, 2, 3, 2), x, K)


def test_degenerate_zero_constant_is_rejected():
    # p = (1,3,3,3), q = -2 n^6 converges to 0, so (0, 0, -8, -10) holds trivially; the scan must reject it
    from scripts.scan_apery_family import accept
    p, q = (1, 3, 3, 3), (0, 0, 0, 0, 0, 0, -2)
    assert accept(p, q, (0, 0, -8, -10)) is None
    # the genuine Apery relation is accepted
    assert accept((5, 27, 51, 34), (0, 0, 0, 0, 0, 0, -1), (6, 0, 0, -1)) is not None
