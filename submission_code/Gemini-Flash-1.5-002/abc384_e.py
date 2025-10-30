H, W, X = map(int, input().split())
P, Q = map(int, input().split())
S = []
for _ in range(H):
    S.append(list(map(int, input().split())))

P -= 1
Q -= 1
total_strength = S[P][Q]
slimes = [(P, Q, S[P][Q])]
visited = set()
visited.add((P, Q))

while True:
    absorbed = False
    max_strength_increase = 0
    best_slime = None
    for p, q, strength in slimes:
        for dp, dq in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            np, nq = p + dp, q + dq
            if 0 <= np < H and 0 <= nq < W and (np, nq) not in visited:
                if S[np][nq] * X < strength:
                    if S[np][nq] > max_strength_increase:
                        max_strength_increase = S[np][nq]
                        best_slime = (np, nq)
                        absorbed = True
    if absorbed:
        total_strength += max_strength_increase
        slimes.append(best_slime)
        visited.add(best_slime)
    else:
        break

print(total_strength)