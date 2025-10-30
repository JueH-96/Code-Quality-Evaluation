# YOUR CODE HERE
N, Q = map(int, input().split())
queries = list(map(int, input().split()))

A = [0] * N
S = set()

for x in queries:
    # Toggle x in S
    if x in S:
        S.remove(x)
    else:
        S.add(x)
    
    # Add |S| to A[j-1] for each j in S (converting to 0-based indexing)
    size_S = len(S)
    for j in S:
        A[j-1] += size_S

print(*A)