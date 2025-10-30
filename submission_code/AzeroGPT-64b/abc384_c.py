a, b, c, d, e = map(int, input().split())

players = [
    "ABCDE", "BCDE", "ACDE", "ABDE", "ABCE", "ABCD",
    "CDE", "BDE", "ADE", "BCE", "ACE", "BCD",
    "ABE", "ACD", "ABD", "ABC",
    "DE", "CE", "BE", "CD",
    "AE", "BD", "AD", "BC",
    "AC", "AB",
    "E", "D", "C", "B", "A"
]

scores = {
    "ABCDE": a + b + c + d + e,
    "BCDE": b + c + d + e,
    "ACDE": a + c + d + e,
    "ABDE": a + b + d + e,
    "ABCE": a + b + c + e,
    "ABCD": a + b + c + d,
    "CDE": c + d + e,
    "BDE": b + d + e,
    "ADE": a + d + e,
    "BCE": b + c + e,
    "ACE": a + c + e,
    "BCD": b + c + d,
    "ABE": a + b + e,
    "ACD": a + c + d,
    "ABD": a + b + d,
    "ABC": a + b + c,
    "DE": d + e,
    "CE": c + e,
    "BE": b + e,
    "CD": c + d,
    "AE": a + e,
    "BD": b + d,
    "AD": a + d,
    "BC": b + c,
    "AC": a + c,
    "AB": a + b,
    "E": e,
    "D": d,
    "C": c,
    "B": b,
    "A": a
}

scores = sorted([(scores[player], player) for player in players], reverse=True)

for score, player in scores:
    print(player)