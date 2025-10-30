import sys
import math

def solve():
    N = int(sys.stdin.readline().strip())
    P = list(map(int, sys.stdin.readline().strip().split()))

    P.sort(reverse=True)

    total_sum = sum(P)
    rank_sum = sum(range(1, N + 1))

    R = (total_sum / rank_sum) - (1200 / math.sqrt(N))

    print("%.6f" % R)

solve()