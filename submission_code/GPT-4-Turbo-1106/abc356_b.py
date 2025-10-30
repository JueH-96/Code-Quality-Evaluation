# Read N and M from the input
N, M = map(int, input().split())

# Read the goal for each nutrient
goals = list(map(int, input().split()))

# Initialize a list to keep track of the total units taken for each nutrient
totals = [0] * M

# Read the units of nutrients taken from each food and update the totals
for _ in range(N):
    nutrients = list(map(int, input().split()))
    for j in range(M):
        totals[j] += nutrients[j]

# Check if the goal is met for all nutrients
if all(totals[i] >= goals[i] for i in range(M)):
    print("Yes")
else:
    print("No")