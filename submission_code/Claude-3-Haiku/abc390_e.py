import sys

# Read input
N, X = map(int, input().split())
foods = []
for _ in range(N):
    V, A, C = map(int, input().split())
    foods.append((V, A, C))

# Initialize the maximum possible value of the minimum intake among vitamins 1, 2, and 3
max_min_intake = 0

# Try all possible subsets of foods
for mask in range(1 << N):
    total_calories = 0
    vitamin_intakes = [0, 0, 0]
    for i in range(N):
        if mask & (1 << i):
            v, a, c = foods[i]
            total_calories += c
            vitamin_intakes[v - 1] += a
            
    # Check if the total calories are within the limit
    if total_calories <= X:
        # Update the maximum possible value of the minimum intake among vitamins 1, 2, and 3
        max_min_intake = max(max_min_intake, min(vitamin_intakes))

# Print the answer
print(max_min_intake)