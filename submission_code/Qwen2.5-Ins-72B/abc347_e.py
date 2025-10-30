# YOUR CODE HERE
N, Q = map(int, input().split())
A = [0] * N
S = set()

for _ in range(Q):
    x = int(input())
    if x in S:
        S.remove(x)
    else:
        S.add(x)
    for j in S:
        A[j-1] += len(S)

print(*A)