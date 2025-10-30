# YOUR CODE HERE
# Read N and M
N, M = map(int, input().split())

# Read goals
A = list(map(int, input().split()))

# Calculate total nutrients consumed
total_nutrients = [0] * M
for i in range(N):
    X_i = list(map(int, input().split()))
    for j in range(M):
        total_nutrients[j] += X_i[j]

# Check if all goals are met
all_goals_met = True
for j in range(M):
    if total_nutrients[j] < A[j]:
        all_goals_met = False
        break

# Output result
if all_goals_met:
    print("Yes")
else:
    print("No")