N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

C = [0]*N
for i in range(M):
    C[B[i]] += 1
    A[B[i]] = 0

D = [0]*N
for i in range(N):
    D[(i+C[i])%N] += A[i]

for i in range(N):
    A[i] += D[i]

print(*A)