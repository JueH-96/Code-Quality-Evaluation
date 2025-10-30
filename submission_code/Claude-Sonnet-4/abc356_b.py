# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Initialize total nutrients consumed to 0
total_nutrients = [0] * M

# Read each food and add its nutrients to the total
for i in range(N):
    nutrients_from_food = list(map(int, input().split()))
    for j in range(M):
        total_nutrients[j] += nutrients_from_food[j]

# Check if all nutrient goals are met
all_goals_met = True
for j in range(M):
    if total_nutrients[j] < A[j]:
        all_goals_met = False
        break

if all_goals_met:
    print("Yes")
else:
    print("No")