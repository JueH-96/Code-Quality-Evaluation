M = int(input())
S = [input() for _ in range(3)]

def check(t):
    c = S[0][t % M]
    for i in range(1, 3):
        if S[i][(t + i) % M] != c:
            return False
    return True

ans = float('inf')
for t in range(M):
    if check(t):
        ans = min(ans, t + 2 * M)

if ans == float('inf'):
    print(-1)
else:
    print(ans)