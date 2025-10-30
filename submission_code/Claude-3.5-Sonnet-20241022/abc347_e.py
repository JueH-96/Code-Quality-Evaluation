N, Q = map(int, input().split())
X = list(map(int, input().split()))

A = [0] * N  # sequence
S = set()    # set of numbers

for x in X:
    if x in S:
        S.remove(x)
    else:
        S.add(x)
    
    # Add |S| to each A_j where j is in S
    for j in S:
        A[j-1] += len(S)

print(*A)