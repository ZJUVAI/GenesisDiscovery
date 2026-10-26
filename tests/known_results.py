"""Benchmark of known identities and inequalities (plan 12, M6).

ALGEBRA entries (name, kind, text): kind "ineq" means the text is >= 0 on positive a, b, c;
kind "eq" means it is 0 identically. CONSTANTS entries (name, residual): residual(dps) is 0 to dps
digits. Each entry has its own test in tests/test_known_results.py. A failing check means the
fixture is wrong: stop and report it; do not loosen the tolerance.
"""
import mpmath as mp

ALGEBRA = [
    ("sum a^2 >= sum ab", "ineq", "sym[a^2] - sym[a*b]"),
    ("sum_sym a^3b >= 2 sum a^2b^2", "ineq", "sym[a^3*b] - 2*sym[a^2*b^2]"),
    ("sum a^3 >= 3abc (AM-GM)", "ineq", "sym[a^3] - 3*a*b*c"),
    ("Schur t=1", "ineq", "sym[a^3] - sym[a^2*b] + 3*a*b*c"),
    ("sum_sym a^2b >= 6abc", "ineq", "sym[a^2*b] - 6*a*b*c"),
    ("Schur t=2 (degree 4)", "ineq", "sym[a^4] - sym[a^3*b] + sym[a^2*b*c]"),
    ("sum a^4 >= sum a^2b^2", "ineq", "sym[a^4] - sym[a^2*b^2]"),
    ("sum a^2b^2 >= abc sum a", "ineq", "sym[a^2*b^2] - sym[a^2*b*c]"),
    ("sum a^3 >= (1/2) sum_sym a^2b", "ineq", "sym[a^3] - sym[a^2*b]/2"),
    ("3 sum a^3 >= (sum a)(sum ab)", "ineq", "3*sym[a^3] - (a+b+c)*sym[a*b]"),
    ("(sum a)^3 >= 27abc", "ineq", "(a+b+c)^3 - 3*3*3*a*b*c"),
    ("sum of six ratios a/b, a/c, ... >= 6", "ineq", "cyc[a/b] + cyc[a/c] - 6"),
    ("Nesbitt", "ineq", "2*cyc[a/(b+c)] - 3"),
    ("Vasc: (sum a^2)^2 >= 3 sum a^3 b", "ineq", "sym[a^2]^2 - 3*cyc[a^3*b]"),
    ("Titu: sum a^2/(b+c) >= (sum a)/2", "ineq", "cyc[a^2/(b+c)] - (a+b+c)/2"),
    ("sum a/b >= 3", "ineq", "cyc[a/b] - 3"),
    ("sum a^2/b >= sum a", "ineq", "cyc[a^2/b] - (a+b+c)"),
    ("sum_cyc ab^2 >= 3abc", "ineq", "cyc[a*b^2] - 3*a*b*c"),
    ("(sum a^2)(sum a) >= 9abc", "ineq", "(a^2+b^2+c^2)*(a+b+c) - 9*a*b*c"),
    ("sum a^3/b >= sum a^2", "ineq", "cyc[a^3/b] - sym[a^2]"),
    ("sum a/(a+b) >= 1", "ineq", "cyc[a/(a+b)] - 1"),
    ("sum (a+b)/(b+c) >= 3", "ineq", "cyc[(a+b)/(b+c)] - 3"),
    ("Titu: sum a^2/(a+b) >= (sum a)/2", "ineq", "cyc[a^2/(a+b)] - (a+b+c)/2"),
    ("identity (sum a)^2 = sum a^2 + 2 sum ab", "eq", "(a+b+c)^2 - sym[a^2] - 2*sym[a*b]"),
    ("identity (sum a)^3", "eq", "(a+b+c)^3 - sym[a^3] - 3*sym[a^2*b] - 6*a*b*c"),
    ("identity (sum a)(sum ab) = sum_sym a^2b + 3abc", "eq",
     "(a+b+c)*sym[a*b] - sym[a^2*b] - 3*a*b*c"),
]


def _euler_e(dps: int) -> mp.mpf:
    """e = 2 + 1/(1 + 1/t_2), t_n = n + n/t_{n+1} (Euler's CF), by backward recurrence."""
    with mp.workdps(dps + 20):
        t = mp.mpf(2 * dps)
        for n in range(2 * dps - 1, 1, -1):
            t = n + n / t
        return 2 + 1 / (1 + 1 / t)


def _pi_lambert(dps: int) -> mp.mpf:
    """pi = 4/K, K = 1 + 1/(3 + 4/(5 + 9/(7 + ...))): a_n = 2n+1, b_n = n^2 (plan M3)."""
    with mp.workdps(dps + 20):
        depth = 4 * dps
        t = mp.mpf(2 * depth + 1)
        for n in range(depth, 0, -1):
            t = (2 * n + 1) + (n + 1) ** 2 / t
        return 4 / (1 + 1 / t)


def _apery_zeta3(dps: int) -> mp.mpf:
    """zeta(3) = 6/(5 - 1^6/(117 - 2^6/(535 - ...))): a(n) = 34n^3+51n^2+27n+5, b(n) = -n^6 (degree 6)."""
    with mp.workdps(dps + 20):
        depth = 2 * dps
        t = mp.mpf(34 * depth ** 3 + 51 * depth ** 2 + 27 * depth + 5)
        for n in range(depth - 1, -1, -1):
            t = (34 * n ** 3 + 51 * n ** 2 + 27 * n + 5) - (n + 1) ** 6 / t
        return mp.mpf(6) / t


def _sqrt2(dps: int) -> mp.mpf:
    """sqrt 2 = 1 + 1/(2 + 1/(2 + ...)) by backward recurrence."""
    with mp.workdps(dps + 20):
        t = mp.mpf(2)
        for _ in range(4 * dps):
            t = 2 + 1 / t
        return 1 + 1 / t


def _phi(dps: int) -> mp.mpf:
    """Golden ratio = 1 + 1/(1 + 1/(1 + ...)) by backward recurrence."""
    with mp.workdps(dps + 20):
        t = mp.mpf(1)
        for _ in range(4 * dps):
            t = 1 + 1 / t
        return t


CONSTANTS = [
    ("zeta(2) = pi^2/6", lambda d: 6 * mp.zeta(2) - mp.pi ** 2),
    ("zeta(4) = pi^4/90", lambda d: 90 * mp.zeta(4) - mp.pi ** 4),
    ("zeta(6) = pi^6/945", lambda d: 945 * mp.zeta(6) - mp.pi ** 6),
    ("Machin: pi/4 = 4 atan(1/5) - atan(1/239)",
     lambda d: 4 * mp.atan(mp.mpf(1) / 5) - mp.atan(mp.mpf(1) / 239) - mp.pi / 4),
    ("Gamma(1/2)^2 = pi", lambda d: mp.gamma(mp.mpf(1) / 2) ** 2 - mp.pi),
    ("Gamma(3/2) = sqrt(pi)/2", lambda d: mp.gamma(mp.mpf(3) / 2) - mp.sqrt(mp.pi) / 2),
    ("Gamma(1/4) Gamma(3/4) = pi sqrt 2",
     lambda d: mp.gamma(mp.mpf(1) / 4) * mp.gamma(mp.mpf(3) / 4) - mp.pi * mp.sqrt(2)),
    ("log 2 = 2 atanh(1/3)", lambda d: mp.log(2) - 2 * mp.atanh(mp.mpf(1) / 3)),
    ("Catalan = sum (-1)^n/(2n+1)^2",
     lambda d: mp.nsum(lambda n: (-1) ** n / (2 * n + 1) ** 2, [0, mp.inf]) - mp.catalan),
    ("e = sum 1/n!", lambda d: mp.nsum(lambda n: 1 / mp.factorial(n), [0, mp.inf]) - mp.e),
    ("e = Euler continued fraction", lambda d: _euler_e(d) - mp.e),
    ("pi = Lambert CF (a_n=2n+1, b_n=n^2)", lambda d: _pi_lambert(d) - mp.pi),
    ("zeta(3) = sum 1/n^3", lambda d: mp.nsum(lambda n: 1 / n ** 3, [1, mp.inf]) - mp.zeta(3)),
    ("zeta(3) = Apery continued fraction (b_n degree 6)", lambda d: _apery_zeta3(d) - mp.zeta(3)),
    ("sqrt 2 periodic continued fraction", lambda d: _sqrt2(d) - mp.sqrt(2)),
    ("golden ratio continued fraction", lambda d: _phi(d) - (1 + mp.sqrt(5)) / 2),
]
