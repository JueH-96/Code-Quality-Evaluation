import sys

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    max_val = max(A)
    min_val = min(A)

    result = (max_val - min_val) * (N * (N - 1) // 2)

    print(result)

solve()