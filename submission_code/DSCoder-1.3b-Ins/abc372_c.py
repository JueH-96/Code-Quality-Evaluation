import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    queries = [tuple(map(int, sys.stdin.readline().split())) for _ in range(Q)]

    count = 0
    for x, c in queries:
        substring = S[x-1:N-x+1]
        count += substring.count('ABC')
        S = S[:x-1] + c + S[x-1:]
        N += 1

    print(count)

solve()