from itertools import permutations

def solve():
    H, W = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(H)]
    B = [list(map(int, input().split())) for _ in range(H)]
    A.sort()
    B.sort()
    if A != B:
        print(-1)
        return
    ans = float('inf')
    for p in permutations(range(W)):
        B = [list(B[i][j] for j in p) for i in range(H)]
        B.sort()
        cnt = sum(abs(i - p[i]) for i in range(W))
        ans = min(ans, cnt)
    print(ans)

solve()