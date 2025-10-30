# YOUR CODE HERE
import sys

# Function to read list of ints
def read_ints():
    """Reads a line of space-separated integers from stdin and returns a list."""
    return list(map(int, sys.stdin.readline().split()))

def solve():
    """Solves the problem based on standard input."""
    N = int(sys.stdin.readline()) # Number of ingredients
    Q = read_ints() # List of available quantities for each ingredient Q_1, ..., Q_N
    A = read_ints() # List of quantities needed for one serving of dish A: A_1, ..., A_N
    B = read_ints() # List of quantities needed for one serving of dish B: B_1, ..., B_N

    # Calculate the maximum number of servings of dish A possible if we *only* make dish A.
    # This value gives an upper bound for the number of servings of dish A (variable x) 
    # we need to consider in our search.
    max_A_servings = float('inf')
    has_positive_A_req = False # Flag to check if dish A requires any ingredient
    for i in range(N):
        if A[i] > 0: # If ingredient i is required for dish A
            has_positive_A_req = True
            # The number of servings of A is limited by this ingredient to floor(Q[i] / A[i])
            # The overall maximum servings of A is limited by the minimum constraint over all required ingredients.
            max_A_servings = min(max_A_servings, Q[i] // A[i])
    
    # According to problem constraints, there is at least one i such that A[i] >= 1.
    # This guarantees that has_positive_A_req will be True, and max_A_servings will become
    # a finite non-negative integer after the loop.
    if not has_positive_A_req:
         # This case should not happen based on problem constraints.
         # If it were possible, it might imply infinite A servings possible.
         # Handle defensively by setting max servings to 0.
         max_A_servings = 0 
    else:
       # Convert the calculated maximum servings to integer, ensuring it's non-negative.
       # The max(0, ...) handles cases where Q[i] // A[i] might somehow be negative, although
       # Q[i] >= 1 and A[i] > 0 ensures non-negativity. It's primarily to safely convert from float('inf').
       max_A_servings = max(0, int(max_A_servings))


    max_total_servings = 0 # Initialize the maximum total servings found so far

    # Iterate through all possible numbers of servings for dish A (let's call it 'x').
    # The range is from 0 up to the maximum possible calculated above (max_A_servings).
    for x in range(max_A_servings + 1):
        
        # For a fixed number of servings 'x' of dish A, calculate the maximum number 
        # of servings 'y' of dish B that can be made with the remaining ingredients.
        current_max_y = float('inf') # Initialize max y for this x to infinity (placeholder for minimum)
        has_positive_B_req = False # Flag to check if dish B requires any ingredient
        
        # Check constraints for dish B based on remaining ingredients
        for i in range(N):
             # Calculate the remaining quantity of ingredient i after making x servings of A.
             # The loop condition `x <= max_A_servings` ensures that A[i] * x <= Q[i] for all i where A[i] > 0.
             # If A[i] == 0, then A[i] * x = 0 <= Q[i] is also true. So remaining_Q >= 0.
             remaining_Q = Q[i] - A[i] * x
             
             # Check if ingredient i is needed for dish B.
             if B[i] > 0:
                 has_positive_B_req = True # Mark that dish B requires at least one ingredient.
                 
                 # If remaining quantity is negative (should not happen due to loop bound), we can make 0 B servings.
                 if remaining_Q < 0:
                     # This should not happen if max_A_servings is calculated correctly.
                     # But as a safeguard: if remaining is negative, 0 servings of B possible based on this ingredient.
                     current_max_y = 0
                     break # No need to check other ingredients, this x is impossible for B

                 # Calculate how many servings of B can be made with the remaining quantity of ingredient i.
                 # Integer division gives the maximum number of full servings.
                 possible_y_for_ingredient = remaining_Q // B[i]
                 
                 # The overall maximum servings of B (`current_max_y`) is limited by the minimum constraint 
                 # over all ingredients required for B.
                 current_max_y = min(current_max_y, possible_y_for_ingredient)

        # After checking all ingredients:
        # Problem constraints guarantee at least one B[i] >= 1, so has_positive_B_req will be True.
        # `current_max_y` will be a finite non-negative integer value unless some error occurred.
        if not has_positive_B_req:
             # This path should not be reachable due to constraints.
             # If it were reached, it might imply infinite B servings are possible.
             pass # Constraints guarantee this won't happen.

        # If `current_max_y` is still infinity, it means B requires no ingredients (all B[i] = 0).
        # This contradicts constraints. Or it could indicate an error.
        if current_max_y == float('inf'):
            # Handle this unexpected state. Defaulting to 0 servings of B.
            current_max_y = 0 
        else:
            # Ensure `current_max_y` is a non-negative integer.
            current_max_y = max(0, int(current_max_y))

        # Update the overall maximum total servings (x + y) found so far.
        max_total_servings = max(max_total_servings, x + current_max_y)

    # Print the final result: the maximum total number of servings possible.
    print(max_total_servings)

# Execute the solve function to run the logic.
solve()