import sys

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    score = 0
    for i in range(2**N):
        B = [(j & i) >> j for j in range(N)]
        if len(set(B)) % M == 0:
            score += sum(B) ** K
        else:
            score += 0
    score %= 998244353

    print(score)

solve()