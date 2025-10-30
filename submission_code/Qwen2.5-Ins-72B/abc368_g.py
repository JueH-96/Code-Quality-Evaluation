# YOUR CODE HERE
import sys

def process_queries(N, A, B, Q, queries):
    results = []
    for query in queries:
        if query[0] == 1:
            i, x = query[1], query[2]
            A[i-1] = x
        elif query[0] == 2:
            i, x = query[1], query[2]
            B[i-1] = x
        elif query[0] == 3:
            l, r = query[1], query[2]
            v = 0
            for i in range(l-1, r):
                v = max(v + A[i], v * B[i])
            results.append(v)
    return results

N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
Q = int(input())
queries = [list(map(int, input().split())) for _ in range(Q)]

results = process_queries(N, A, B, Q, queries)
for result in results:
    print(result)