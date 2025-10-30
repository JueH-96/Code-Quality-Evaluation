import sys

def solve():
    N, Q = map(int, input().split())
    A = [0] * N
    S = set()
    queries = list(map(int, input().split()))

    for x in queries:
        if x in S:
            S.remove(x)
        else:
            S.add(x)
        for j in range(1, N + 1):
            if j in S:
                A[j - 1] += len(S)

    print(*A)

if __name__ == "__main__":
    solve()