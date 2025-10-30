# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
taken = [0] * M
for _ in range(N):
    X = list(map(int, input().split()))
    for j in range(M):
        taken[j] += X[j]

for j in range(M):
    if taken[j] < A[j]:
        print("No")
        exit()
print("Yes")