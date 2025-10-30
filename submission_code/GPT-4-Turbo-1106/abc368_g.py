def process_queries(N, A, B, queries):
    for query in queries:
        q = query.split()
        if q[0] == '1':
            A[int(q[1]) - 1] = int(q[2])
        elif q[0] == '2':
            B[int(q[1]) - 1] = int(q[2])
        elif q[0] == '3':
            l, r = int(q[1]) - 1, int(q[2])
            v = 0
            for i in range(l, r):
                v = max(v + A[i], v * B[i])
            print(v)

# Read input
N = int(input().strip())
A = list(map(int, input().strip().split()))
B = list(map(int, input().strip().split()))
Q = int(input().strip())
queries = [input().strip() for _ in range(Q)]

# Process queries
process_queries(N, A, B, queries)