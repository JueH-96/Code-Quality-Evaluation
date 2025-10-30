import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    R = list(map(int, sys.stdin.readline().split()))
    queries = [int(sys.stdin.readline()) for _ in range(Q)]

    R.sort(reverse=True)

    for query in queries:
        count = 0
        for r in R:
            if r <= query:
                query -= r
                count += 1
        print(count)

if __name__ == "__main__":
    solve()