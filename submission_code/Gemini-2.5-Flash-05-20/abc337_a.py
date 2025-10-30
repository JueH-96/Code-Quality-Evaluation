# YOUR CODE HERE
import sys

def solve():
    N = int(sys.stdin.readline())

    takahashi_total_score = 0
    aoki_total_score = 0

    for _ in range(N):
        X, Y = map(int, sys.stdin.readline().split())
        takahashi_total_score += X
        aoki_total_score += Y

    if takahashi_total_score > aoki_total_score:
        print("Takahashi")
    elif aoki_total_score > takahashi_total_score:
        print("Aoki")
    else:
        print("Draw")

solve()