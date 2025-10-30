# Read input values
N, M = map(int, input().split())
A = list(map(int, input().split()))
X = [list(map(int, input().split())) for _ in range(N)]

# Initialize a list to store the total intake of each nutrient
total = [0] * M

# Calculate the total intake for each nutrient
for i in range(N):
    for j in range(M):
        total[j] += X[i][j]

# Check if all nutrients meet the goal
meets_goal = True
for j in range(M):
    if total[j] < A[j]:
        meets_goal = False
        break

# Output the result
if meets_goal:
    print("Yes")
else:
    print("No")