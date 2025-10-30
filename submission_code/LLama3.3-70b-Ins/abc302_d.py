import sys

def solve():
    N, M, D = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    max_sum = -1
    for a in A:
        for b in B:
            if abs(a - b) <= D:
                max_sum = max(max_sum, a + b)

    print(max_sum)

if __name__ == "__main__":
    solve()