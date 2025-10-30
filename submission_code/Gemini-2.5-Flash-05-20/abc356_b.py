import sys

def solve():
    # Read N (number of foods) and M (number of nutrient types)
    N, M = map(int, sys.stdin.readline().split())

    # Read the goal amounts for each nutrient (A_1 to A_M)
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize a list to store the total intake for each nutrient.
    # The size of this list will be M, and all elements start at 0.
    total_nutrient_intake = [0] * M

    # Read the nutrient intake for each food
    for _ in range(N):
        # Read the M units of nutrients from the current food
        food_nutrients = list(map(int, sys.stdin.readline().split()))

        # Add the nutrients from the current food to the running totals
        for j in range(M):
            total_nutrient_intake[j] += food_nutrients[j]

    # Check if the total intake for each nutrient meets its respective goal
    all_goals_met = True
    for j in range(M):
        # If the total intake for nutrient j is less than its goal A[j],
        # then the goal is not met for this nutrient.
        if total_nutrient_intake[j] < A[j]:
            all_goals_met = False
            break # No need to check further, since at least one goal is not met

    # Print the result based on whether all goals were met
    if all_goals_met:
        print("Yes")
    else:
        print("No")

# Call the solve function to run the program
solve()