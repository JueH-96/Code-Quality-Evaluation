def check_index(i, j):
    return i >= 0 and i <= H and j >= 0 and j <= W

INF = 10 ** 18
H, W, X = map(int, input().split())
P, Q = map(lambda x: int(x) - 1, input().split())
S = [[int(i) for i in input().split()] for _ in range(H)]

stack = [(P, Q)]

answer = S[P][Q]
history = [[False] * W for _ in range(H)]
while stack:
    P, Q = stack.pop()
    if history[P][Q]:
        continue
    history[P][Q] = True    

    for dP, dQ in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
        nextP, nextQ = P + dP, Q + dQ
        if not check_index(nextP, nextQ):
            continue
        if history[nextP][nextQ]:
            continue
        next_cell = S[nextP][nextQ]
        if next_cell < answer / X:
            answer += next_cell
            stack.append((nextP, nextQ))
print(answer)