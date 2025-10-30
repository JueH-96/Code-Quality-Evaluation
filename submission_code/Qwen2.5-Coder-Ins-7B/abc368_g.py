# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
B = list(map(int, data[N+1:2*N+1]))
Q = int(data[2*N+1])

queries = []
index = 2*N+2
for _ in range(Q):
    query = list(map(int, data[index:index+3]))
    queries.append(query)
    index += 3

def process_queries(N, A, B, queries):
    results = []
    for query in queries:
        if query[0] == 1:
            A[query[1]-1] = query[2]
        elif query[0] == 2:
            B[query[1]-1] = query[2]
        elif query[0] == 3:
            l, r = query[1]-1, query[2]
            v = 0
            for i in range(l, r):
                v = max(v + A[i], v * B[i])
            results.append(v)
    return results

results = process_queries(N, A, B, queries)
for result in results:
    print(result)