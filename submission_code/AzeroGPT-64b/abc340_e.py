N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

L = [-1]*N # The last operation which modifies the ith box
T = [-1]*N # The number of balls in the ith box
for i in range(N):
    T[i] = A[i]

for i in range(M):
    B[i] = (N + B[i]) % N

ans = [0]*N

for i, Bi in enumerate(B):
    C = 0
    while (n := T[B[i] + C]):
        L[(B[i] + C) % N] = i
        ans[(B[i] + C) % N] = n
        C += 1
        T[(B[i] + C) % N] += 1

for i in range(N):
    if L[i] == M-1 and ans[i] == 0:
        ans[i] = T[i]
print(*ans)