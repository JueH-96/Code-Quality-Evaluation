# YOUR CODE HERE
n, q = map(int, input().split())
queries = list(map(int, input().split()))

A = [0] * (n + 1)  # 1-indexed
S = set()

for x in queries:
    if x in S:
        S.remove(x)
    else:
        S.add(x)
    
    size = len(S)
    for j in S:
        A[j] += size

print(' '.join(map(str, A[1:])))