import sys

# Read the inputs
N, M = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
X = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]

# Initialize a list to store the total amount of each nutrient taken
total_nutrients = [0] * M

# Calculate the total amount of each nutrient taken
for i in range(N):
    for j in range(M):
        total_nutrients[j] += X[i][j]

# Check if the goal is met for all nutrients
for i in range(M):
    if total_nutrients[i] < A[i]:
        print("No")
        sys.exit()

print("Yes")