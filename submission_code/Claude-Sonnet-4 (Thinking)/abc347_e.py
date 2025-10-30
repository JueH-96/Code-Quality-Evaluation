N, Q = map(int, input().split())
queries = list(map(int, input().split()))

A = [0] * (N + 1)  # A[0] is unused, A[1] to A[N] are used
S = set()

for x in queries:
    if x in S:
        S.remove(x)
    else:
        S.add(x)
    
    # Add |S| to A_j for each j in S
    for j in S:
        A[j] += len(S)

# Output A[1] to A[N]
print(' '.join(map(str, A[1:N+1])))