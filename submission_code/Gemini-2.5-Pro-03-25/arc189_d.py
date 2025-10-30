# YOUR CODE HERE
import sys

def compute_max_size(K, N, A):
    """
    Computes the maximum size Takahashi can achieve starting as the K-th slime (0-indexed).
    
    Args:
        K (int): The 0-based index of the starting slime (Takahashi).
        N (int): The total number of slimes.
        A (list): A list containing the sizes of the slimes (0-indexed).

    Returns:
        int: The maximum possible size Takahashi can achieve.
    """
    
    # Initialize the boundaries of the merged slime containing Takahashi.
    # Initially, it only contains Takahashi himself.
    current_L = K # Left boundary (inclusive)
    current_R = K # Right boundary (inclusive)
    
    # Initialize the total size of the merged slime.
    # Initially, it's just the size of the K-th slime.
    current_S = A[K] 

    # Loop indefinitely until no more absorptions are possible in a full pass.
    while True:
        
        # Flag to track if any absorption (expansion) happened in the current pass.
        progressed = False 
        
        # Check if Takahashi can absorb the slime to his left.
        # Condition 1: Check index bounds. There must be a slime to the left (current_L > 0).
        # Condition 2: Check value condition. The size of the left neighbor (A[current_L - 1]) 
        # must be strictly less than Takahashi's current total size (current_S).
        if current_L > 0 and A[current_L - 1] < current_S:
            # If absorbable, expand the merged slime's range to the left.
            current_L -= 1 
            # Increase Takahashi's total size by the size of the absorbed slime.
            current_S += A[current_L] 
            # Mark that progress was made in this pass.
            progressed = True 

        # Check if Takahashi can absorb the slime to his right.
        # Condition 1: Check index bounds. There must be a slime to the right (current_R < N - 1).
        # Condition 2: Check value condition. The size of the right neighbor (A[current_R + 1])
        # must be strictly less than Takahashi's current total size (current_S).
        if current_R < N - 1 and A[current_R + 1] < current_S:
            # If absorbable, expand the merged slime's range to the right.
            current_R += 1 
            # Increase Takahashi's total size by the size of the absorbed slime.
            current_S += A[current_R] 
            # Mark that progress was made in this pass.
            progressed = True 

        # If no slime was absorbed (neither left nor right expansion occurred) in this pass,
        # it means Takahashi cannot grow any larger. The process terminates.
        if not progressed:
            break
            
    # Return the final maximum size achieved.
    return current_S

# Main solver function orchestrating the overall process.
def solve():
    # Read the number of slimes, N.
    N = int(sys.stdin.readline())
    # Read the list of slime sizes, A.
    A = list(map(int, sys.stdin.readline().split()))

    # Create a list to store the computed maximum size for each possible starting slime K.
    results = []
    
    # Iterate through each possible starting position K for Takahashi.
    # The problem uses 1-based indexing (K=1..N), but Python uses 0-based indexing.
    # So we iterate K from 0 to N-1.
    for K in range(N): 
        # Compute the maximum size for starting slime K using the helper function.
        max_size = compute_max_size(K, N, A)
        # Append the result to the list.
        results.append(max_size)

    # Print all the computed results, separated by spaces.
    # The * operator unpacks the list `results` into separate arguments for print().
    print(*(results))

# Execute the solver function to run the program.
solve()

# END OF YOUR CODE HERE