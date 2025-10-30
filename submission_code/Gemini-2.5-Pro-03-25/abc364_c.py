# YOUR CODE HERE
import sys

# Function to read input and solve the problem
def solve():
    # Read N, X, Y from the first line
    # N: number of dishes
    # X: maximum total sweetness allowed (exclusive threshold)
    # Y: maximum total saltiness allowed (exclusive threshold)
    N, X, Y = map(int, sys.stdin.readline().split())
    
    # Read A_1, ..., A_N (sweetness values) from the second line
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read B_1, ..., B_N (saltiness values) from the third line
    B = list(map(int, sys.stdin.readline().split()))

    # Create a list of dictionaries, where each dictionary represents a dish
    # Each dish has 'A' (sweetness) and 'B' (saltiness) properties.
    # Storing A and B values together simplifies sorting based on either key later.
    dishes = []
    for i in range(N):
        dishes.append({'A': A[i], 'B': B[i]})

    # Calculate total sweetness and saltiness if all dishes were eaten.
    # This is used for an optimization: if total sums don't exceed limits,
    # Takahashi will eat all N dishes regardless of the order.
    total_A = sum(A) # Use sum() for potentially faster calculation than manual loop
    total_B = sum(B)
    
    # Check if Takahashi can eat all dishes without exceeding the limits.
    # The condition to stop is strictly greater (> X or > Y).
    # If total sums are less than or equal to thresholds, he eats all N dishes.
    if total_A <= X and total_B <= Y:
        print(N)
        return

    # If either total sum exceeds its limit, Takahashi must stop eating 
    # at some point before or at N dishes. We need to find the minimum 
    # possible number of dishes he eats across all possible arrangements.
    # The logic derived shows that this minimum is achieved either by
    # sorting dishes by sweetness (A) descending or by saltiness (B) descending.
    # We calculate the number of dishes eaten for these two specific arrangements
    # and take the minimum.

    # Calculate k_A: number of dishes eaten when the arrangement is sorted by A descending.
    # Sort dishes based on sweetness A in descending order.
    # Python's sort is stable, but stability doesn't affect the result here.
    dishes_sorted_A = sorted(dishes, key=lambda d: d['A'], reverse=True)
    
    current_A = 0 # Tracks cumulative sweetness
    current_B = 0 # Tracks cumulative saltiness
    k_A = 0 # Counts dishes eaten
    # Iterate through the dishes sorted by sweetness descending
    for i in range(N):
        # Add the current dish's sweetness and saltiness
        current_A += dishes_sorted_A[i]['A']
        current_B += dishes_sorted_A[i]['B']
        k_A += 1 # Increment dish count
        # Check if either limit is exceeded
        if current_A > X or current_B > Y:
            # If a limit is exceeded, Takahashi stops eating. Break the loop.
            break 
    # After the loop, k_A holds the number of dishes eaten in this arrangement.
    # Since we already checked that total sums exceed limits, this loop is guaranteed
    # to break at some point where k_A <= N.

    # Calculate k_B: number of dishes eaten when the arrangement is sorted by B descending.
    # Sort dishes based on saltiness B in descending order.
    dishes_sorted_B = sorted(dishes, key=lambda d: d['B'], reverse=True)
    
    current_A = 0 # Reset cumulative sums
    current_B = 0
    k_B = 0 # Reset dish count
    # Iterate through the dishes sorted by saltiness descending
    for i in range(N):
        # Add the current dish's sweetness and saltiness
        current_A += dishes_sorted_B[i]['A']
        current_B += dishes_sorted_B[i]['B']
        k_B += 1 # Increment dish count
        # Check if either limit is exceeded
        if current_A > X or current_B > Y:
            # If a limit is exceeded, Takahashi stops eating. Break the loop.
            break
    # After the loop, k_B holds the number of dishes eaten in this arrangement.
    # Similar to k_A, this loop is guaranteed to break with k_B <= N.

    # The minimum possible number of dishes Takahashi eats is the minimum of
    # the counts obtained from the two greedy strategies (sorting by A desc, sorting by B desc).
    print(min(k_A, k_B))

# Call the solve function to execute the logic
solve()