# YOUR CODE HERE
# Read N and M
N, M = map(int, input().split())

# Read goals for each nutrient
goals = list(map(int, input().split()))

# Initialize totals for each nutrient
totals = [0] * M

# Read nutrient content for each food and accumulate totals
for i in range(N):
    nutrients = list(map(int, input().split()))
    for j in range(M):
        totals[j] += nutrients[j]

# Check if all goals are met
all_goals_met = True
for j in range(M):
    if totals[j] < goals[j]:
        all_goals_met = False
        break

# Output result
if all_goals_met:
    print("Yes")
else:
    print("No")