# YOUR CODE HERE
N, Q = map(int, input().split())
queries = list(map(int, input().split()))

A = [0] * N
S = set()

for x in queries:
    if x in S:
        S.remove(x)
    else:
        S.add(x)
    
    size = len(S)
    for j in S:
        A[j-1] += size

print(' '.join(map(str, A)))