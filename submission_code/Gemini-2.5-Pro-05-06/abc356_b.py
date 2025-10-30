# Read N and M, the number of foods and number of nutrients
N, M = map(int, input().split())

# Read A, the list of target nutrient amounts (M integers)
A = list(map(int, input().split()))

# Initialize `total_nutrients_taken` as a list of M zeros.
# `total_nutrients_taken[j]` will store the total amount of nutrient j consumed.
total_nutrients_taken = [0] * M

# Loop N times, once for each food
for _ in range(N):
    # Read X_row, a list of M integers representing nutrient amounts from the current food
    X_row = list(map(int, input().split()))
    # Iterate through each nutrient type for the current food
    for j in range(M):
        # Add the amount of nutrient j from this food to the total for nutrient j
        total_nutrients_taken[j] += X_row[j]

# Check if all nutrient goals are met.
# We use a generator expression within the `all()` function.
# For each nutrient j (from 0 to M-1):
#   Check if `total_nutrients_taken[j]` is greater than or equal to `A[j]`.
# `all()` returns True if all these conditions are True, False otherwise.
# `all()` also short-circuits, meaning it stops checking as soon as one False condition is found.
if all(total_nutrients_taken[j] >= A[j] for j in range(M)):
    print("Yes")
else:
    print("No")