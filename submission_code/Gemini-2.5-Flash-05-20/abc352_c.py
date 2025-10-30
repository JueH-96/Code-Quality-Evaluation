import sys

def solve():
    # Read N, the number of giants
    N = int(sys.stdin.readline())

    # total_A will accumulate the sum of all A_i values.
    # This sum forms the base of the total height.
    total_A = 0

    # max_B_minus_A will store the maximum difference (B_i - A_i) found among all giants.
    # This difference represents the 'extra' height contributed by the topmost giant's head
    # above its own shoulder level. Since A_i <= B_i, B_i - A_i is always non-negative,
    # so initializing with 0 is correct.
    max_B_minus_A = 0 

    # Iterate N times to read the A_i and B_i values for each giant
    for _ in range(N):
        A, B = map(int, sys.stdin.readline().split())
        
        # Add the current giant's shoulder height (A) to the total sum of A's
        total_A += A
        
        # Calculate the difference between head height and shoulder height for the current giant
        current_diff = B - A
        
        # Update max_B_minus_A if the current_diff is greater than the previously found maximum
        max_B_minus_A = max(max_B_minus_A, current_diff)

    # The maximum possible height of the topmost giant's head is derived as:
    # H_N = (Sum of A values of P_1 to P_{N-1}) + B_{P_N}
    # We know that (Sum of A values of P_1 to P_{N-1}) = (Sum of all A_i values) - A_{P_N}
    # So, H_N = (total_A - A_{P_N}) + B_{P_N}
    # H_N = total_A + (B_{P_N} - A_{P_N})
    # To maximize H_N, we need to maximize (B_{P_N} - A_{P_N}) because total_A is constant.
    # This is achieved by selecting the giant with the largest (B_i - A_i) as P_N.
    
    result = total_A + max_B_minus_A
    
    # Print the final calculated maximum height
    print(result)

# Call the solve function to execute the program logic
solve()