N, M = map(int, input().split())
A = list(map(int, input().split()))
total_nutrients = [0] * M

for _ in range(N):
    X = list(map(int, input().split()))
    for j in range(M):
        total_nutrients[j] += X[j]

if all(total_nutrients[j] >= A[j] for j in range(M)):
    print("Yes")
else:
    print("No")