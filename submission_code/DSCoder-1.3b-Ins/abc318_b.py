# YOUR CODE HERE

N = int(input())
A, B, C, D = [], [], [], []

for _ in range(N):
    a, b, c, d = map(int, input().split())
    A.append(a)
    B.append(b)
    C.append(c)
    D.append(d)

S = 0
for i in range(N):
    S += (B[i] - A[i] + 1) * (D[i] - C[i] + 1)

print(S)