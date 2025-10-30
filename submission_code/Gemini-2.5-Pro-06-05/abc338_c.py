import sys

def solve():
    """
    Reads refrigerator and dish data, then calculates and prints the maximum
    total number of servings possible.
    """
    try:
        # Read problem parameters from standard input
        N = int(sys.stdin.readline())
        Q = list(map(int, sys.stdin.readline().split()))
        A = list(map(int, sys.stdin.readline().split()))
        B = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handle potential empty input or format errors
        return

    # Determine the upper bound for the number of servings of dish A.
    # This is the maximum number we can make if we only make dish A.
    max_a = float('inf')
    for i in range(N):
        if A[i] > 0:
            max_a = min(max_a, Q[i] // A[i])
    
    # As per constraints, at least one A_i > 0, so max_a is a finite integer.

    max_total_servings = 0

    # Iterate through all possible numbers of dish A (num_a) from 0 to max_a.
    for num_a in range(int(max_a) + 1):
        # For the current num_a, find the maximum possible number of dish B (num_b).
        # This is limited by the remaining ingredients.
        max_num_b = float('inf')
        
        for i in range(N):
            # Calculate remaining quantity of ingredient i
            remaining_q = Q[i] - num_a * A[i]
            
            # If B_i > 0, it imposes a constraint on num_b
            if B[i] > 0:
                max_num_b = min(max_num_b, remaining_q // B[i])
        
        # As per constraints, at least one B_i > 0, so max_num_b is a finite integer.
        num_b = int(max_num_b)
        
        # Update the overall maximum total servings.
        max_total_servings = max(max_total_servings, num_a + num_b)

    print(max_total_servings)

solve()