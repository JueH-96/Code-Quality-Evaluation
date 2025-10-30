import sys

def solve():
    N = int(sys.stdin.readline())
    total_takahashi_score = 0
    total_aoki_score = 0

    for _ in range(N):
        X, Y = map(int, sys.stdin.readline().split())
        total_takahashi_score += X
        total_aoki_score += Y

    if total_takahashi_score > total_aoki_score:
        print("Takahashi")
    elif total_aoki_score > total_takahashi_score:
        print("Aoki")
    else:
        print("Draw")

solve()