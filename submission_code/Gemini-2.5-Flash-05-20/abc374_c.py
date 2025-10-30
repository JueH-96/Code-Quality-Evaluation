# YOUR CODE HERE
import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())
    
    # Read K_i values
    K = list(map(int, sys.stdin.readline().split()))

    # Initialize min_max_people to a very large value
    min_max_people = float('inf')

    # Iterate through all 2^N possible ways to divide departments.
    # Each integer 'i' from 0 to (2^N - 1) represents a unique assignment.
    # The j-th bit of 'i' (0 or 1) determines which group the j-th department K[j] belongs to.
    # For example, if bit is 0, assign to Group A; if 1, assign to Group B.
    for i in range(1 << N): # (1 << N) is equivalent to 2^N
        sum_A = 0
        sum_B = 0

        # Iterate through each department
        for j in range(N):
            # Check the j-th bit of 'i'
            if (i >> j) & 1:
                # If the j-th bit is 1, assign K[j] to Group B
                sum_B += K[j]
            else:
                # If the j-th bit is 0, assign K[j] to Group A
                sum_A += K[j]
        
        # Calculate the maximum people for the current assignment
        current_max_people = max(sum_A, sum_B)
        
        # Update the overall minimum of these maximums
        min_max_people = min(min_max_people, current_max_people)
    
    # Print the final result
    print(min_max_people)

# Call the solver function to execute the program
solve()