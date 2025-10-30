# YOUR CODE HERE
import sys

# Read N (number of foods) and M (number of nutrients)
# N, M are given as the first line
N, M = map(int, sys.stdin.readline().split())

# Read the required units for each nutrient A_1, ..., A_M
# This is the second line of input.
# Store in a list A. A[j] corresponds to the requirement for nutrient j+1.
# We use 0-indexing internally, so A has M elements, A[0] to A[M-1].
A = list(map(int, sys.stdin.readline().split()))

# Initialize a list to store the total intake for each nutrient across all foods.
# nutrient_totals[j] will store the sum of X_i,j+1 for all i from 1 to N.
# The list has M elements, 0-indexed, corresponding to nutrient 1 through M.
nutrient_totals = [0] * M

# Read the nutrient intake from each of the N foods.
# The input gives N lines following the nutrient goals.
for i in range(N):
    # Read the M nutrient values for the i-th food (using 0-indexing for the loop counter)
    # X_i+1,1, ..., X_i+1,M are read from the i-th line of food data into a list food_nutrients.
    food_nutrients = list(map(int, sys.stdin.readline().split()))

    # Add the nutrient values of this food to the respective totals.
    # Iterate through each nutrient type from 1 to M (using 0-indexing j from 0 to M-1).
    for j in range(M):
        # food_nutrients[j] is the amount of nutrient j+1 from food i+1 (using problem's 1-indexing).
        # nutrient_totals[j] accumulates the total amount of nutrient j+1 across all foods eaten so far.
        nutrient_totals[j] += food_nutrients[j]

# Check if the goal is met for all M nutrients (nutrient 1 to M).
# Iterate through each nutrient index from 0 to M-1.
for j in range(M):
    # Compare the total intake for nutrient j+1 (nutrient_totals[j])
    # with the required amount for nutrient j+1 (A[j]).
    if nutrient_totals[j] < A[j]:
        # The goal is not met for this specific nutrient (j+1).
        # Since the overall requirement is that ALL goals must be met,
        # we can immediately conclude the overall goal is not met.
        print("No")
        # Exit the program using sys.exit()
        sys.exit()

# If the loop finishes without finding any nutrient deficiency for any nutrient,
# it means the goal was met for all nutrients.
print("Yes")