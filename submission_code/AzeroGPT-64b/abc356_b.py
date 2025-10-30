N, M = map(int, input().split())
A = list(map(int, input().split()))

# Initialize a list to store the total consumption for each nutrient
consumption_per_nutrient = [0] * M

# Iterate through each food consumed
for _ in range(N):
    X = list(map(int, input().split()))
    # Update the total consumption for each nutrient
    for i in range(M):
        consumption_per_nutrient[i] += X[i]

# Check if the goal is met for all nutrients
all_goals_met = all(consumption >= goal for consumption, goal in zip(consumption_per_nutrient, A))

if all_goals_met:
    print("Yes")
else:
    print("No")