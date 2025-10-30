# YOUR CODE HERE
N, Q = map(int, input().split())
queries = list(map(int, input().split()))

A = [0]*N
S = set()

for x in queries:
    if x in S:
        S.remove(x)
    else:
        S.add(x)
    for i in range(N):
        if i+1 in S:
            A[i] += len(S)

print(*A)