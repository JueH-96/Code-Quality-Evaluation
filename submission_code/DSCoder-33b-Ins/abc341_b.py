# YOUR CODE HERE

N = int(input())
A = list(map(int, input().split()))
S = []
T = []
for _ in range(N - 1):
    s, t = map(int, input().split())
    S.append(s)
    T.append(t)

for i in range(N - 2, -1, -1):
    A[i] = min(A[i], A[i + 1] - S[i] + T[i])

print(A[0])