a, b, c, d, e = map(int, input().split())

participants = {
    "A": a,
    "B": b,
    "C": c,
    "D": d,
    "E": e,
    "AB": a + b,
    "AC": a + c,
    "AD": a + d,
    "AE": a + e,
    "BC": b + c,
    "BD": b + d,
    "BE": b + e,
    "CD": c + d,
    "CE": c + e,
    "DE": d + e,
    "ABC": a + b + c,
    "ABD": a + b + d,
    "ABE": a + b + e,
    "ACD": a + c + d,
    "ACE": a + c + e,
    "ADE": a + d + e,
    "BCD": b + c + d,
    "BCE": b + c + e,
    "BDE": b + d + e,
    "CDE": c + d + e,
    "ABCD": a + b + c + d,
    "ABCE": a + b + c + e,
    "ABDE": a + b + d + e,
    "ACDE": a + c + d + e,
    "BCDE": b + c + d + e,
    "ABCDE": a + b + c + d + e
}

scores = []
for name, score in participants.items():
    scores.append((score, name))

scores.sort(key=lambda x: (-x[0], x[1]))

for score, name in scores:
    print(name)