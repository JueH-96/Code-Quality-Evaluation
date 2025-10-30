def process_queries(N, Q, queries):
    A = [0] * N
    S = set()

    for x in queries:
        if x in S:
            S.remove(x)
        else:
            S.add(x)
        
        current_size = len(S)
        for j in S:
            A[j - 1] += current_size

    return A

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
queries = list(map(int, data[2:]))

result = process_queries(N, Q, queries)
print(' '.join(map(str, result)))