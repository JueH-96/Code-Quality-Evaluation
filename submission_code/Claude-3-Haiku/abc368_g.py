import sys

def solve_queries(N, A, B, Q, queries):
    for query in queries:
        query_type, i, x = map(int, query.split())
        if query_type == 1:
            A[i-1] = x
        elif query_type == 2:
            B[i-1] = x
        else:
            l, r = i, x
            v = 0
            for j in range(l-1, r):
                v = max(v + A[j], v * B[j])
            print(v)

if __name__ == "__main__":
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))
    B = list(map(int, sys.stdin.readline().strip().split()))
    Q = int(sys.stdin.readline().strip())
    queries = [sys.stdin.readline().strip() for _ in range(Q)]
    solve_queries(N, A, B, Q, queries)