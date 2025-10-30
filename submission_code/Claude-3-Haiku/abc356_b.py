# Read the input
N, M = map(int, input().split())
A = list(map(int, input().split()))
X = [list(map(int, input().split())) for _ in range(N)]

# Check if the goal is met for all nutrients
for j in range(M):
    total = sum(x[j] for x in X)
    if total < A[j]:
        print("No")
        return

print("Yes")