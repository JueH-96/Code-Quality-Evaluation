import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    A.sort()
    B.sort()

    for x in range(1, 10**9 + 1):
        sellers = sum(1 for a in A if a <= x)
        buyers = sum(1 for b in B if b >= x)
        if sellers >= buyers:
            print(x)
            return

if __name__ == "__main__":
    solve()