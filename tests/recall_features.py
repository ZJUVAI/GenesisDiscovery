"""Feature sets for the degree <= 3 benchmark inequalities (M6 solver recall, plan 12)."""

RECALL_FEATURES = {
    "sum a^2 >= sum ab": ["sym[a^2]", "sym[a*b]"],
    "sum a^3 >= 3abc (AM-GM)": ["sym[a^3]", "a*b*c"],
    "Schur t=1": ["sym[a^3]", "sym[a^2*b]", "a*b*c"],
    "sum_sym a^2b >= 6abc": ["sym[a^2*b]", "a*b*c"],
    "sum a^3 >= (1/2) sum_sym a^2b": ["sym[a^3]", "sym[a^2*b]"],
    "3 sum a^3 >= (sum a)(sum ab)": ["sym[a^3]", "(a+b+c)*sym[a*b]"],
    "(sum a)^3 >= 27abc": ["(a+b+c)^3", "a*b*c"],
    "Nesbitt": ["cyc[a/(b+c)]", "1"],
    "sum a/b >= 3": ["cyc[a/b]", "1"],
    "sum_cyc ab^2 >= 3abc": ["cyc[a*b^2]", "a*b*c"],
    "sum a/(a+b) >= 1": ["cyc[a/(a+b)]", "1"],
    "sum (a+b)/(b+c) >= 3": ["cyc[(a+b)/(b+c)]", "1"],
    "sum a^2/b >= sum a": ["cyc[a^2/b]", "sym[a]"],
    "Titu: sum a^2/(b+c) >= (sum a)/2": ["cyc[a^2/(b+c)]", "sym[a]"],
    "Titu: sum a^2/(a+b) >= (sum a)/2": ["cyc[a^2/(a+b)]", "sym[a]"],
    "(sum a^2)(sum a) >= 9abc": ["(a+b+c)*sym[a^2]", "a*b*c"],
    "sum a^3/b >= sum a^2": ["cyc[a^3/b]", "sym[a^2]"],
}
