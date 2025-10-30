import sys

def solve():
    """
    Reads nutrient data and determines if nutritional goals are met.
    """
    try:
        # Read N (number of foods) and M (number of nutrients)
        n, m = map(int, sys.stdin.readline().split())

        # Read the target amounts A for each of the M nutrients
        A = list(map(int, sys.stdin.readline().split()))

        # Initialize a list to store the total intake for each nutrient.
        # This list will have M elements, all initially set to 0.
        total_intake = [0] * m

        # Loop N times to process each food's nutrient data
        for _ in range(n):
            # Read the nutrient values (X_i,1, ..., X_i,M) for the current food
            food_nutrients = list(map(int, sys.stdin.readline().split()))
            
            # Add the nutrients from this food to the running totals
            for j in range(m):
                total_intake[j] += food_nutrients[j]
    
    except (IOError, ValueError):
        # This handles cases of empty or malformed input, which are not
        # expected based on the problem constraints but are good practice to consider.
        # In a competitive programming context, this might be omitted for speed.
        return

    # Determine if the goal is met for all nutrients.
    # The `zip` function pairs each element of `total_intake` with the corresponding
    # element of `A`.
    # The generator expression `(intake >= required for ...)` checks the condition
    # for each nutrient.
    # The `all()` function returns True only if the condition is met for all nutrients.
    if all(intake >= required for intake, required in zip(total_intake, A)):
        print("Yes")
    else:
        print("No")

# Run the solution
solve()