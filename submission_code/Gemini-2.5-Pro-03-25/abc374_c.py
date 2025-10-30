# YOUR CODE HERE
import sys

def solve():
    # Read input N, the number of departments.
    N = int(sys.stdin.readline())
    
    # Read input K_1, K_2, ..., K_N, the number of people in each department.
    # The input values are space-separated on a single line.
    # map(int, ...) converts each string part into an integer.
    # list(...) converts the map object into a list.
    K = list(map(int, sys.stdin.readline().split()))

    # Calculate the total sum of people across all departments.
    # This sum will be constant regardless of how departments are partitioned.
    total_sum = sum(K)

    # Initialize the variable to store the minimum possible value of the maximum group size.
    # We are looking for the smallest possible maximum.
    # A safe initial upper bound is the total sum itself. This could occur if one group gets all departments.
    min_max_group_sum = total_sum 

    # The problem requires partitioning N departments into two groups, A and B.
    # We can iterate through all possible ways to assign departments to Group A.
    # The remaining departments automatically form Group B.
    # There are 2^N possible subsets of departments that can form Group A.
    # We can represent each subset using a bitmask from 0 to 2^N - 1.
    
    # Calculate 2^N. This is the total number of subsets (including empty set and full set).
    limit = 1 << N 

    # Iterate through each integer `i` from 0 to 2^N - 1.
    # Each value of `i` corresponds to a unique subset using its binary representation.
    # If the j-th bit of `i` is 1, it means the j-th department is included in Group A.
    for i in range(limit):
        # Initialize the sum of people for Group A for the current subset `i`.
        current_sum_A = 0
        
        # Iterate through each department index `j` from 0 to N-1.
        for j in range(N):
            # Check if the j-th bit of `i` is set (equal to 1).
            # `(i >> j)` shifts the bits of `i` to the right by `j` positions.
            # `& 1` checks if the least significant bit (which was originally the j-th bit of `i`) is 1.
            if (i >> j) & 1:
                # If the j-th bit is 1, it means the department corresponding to index `j` 
                # (which has K[j] people) is assigned to Group A for this partition.
                # Add its population to Group A's total.
                current_sum_A += K[j]
        
        # Calculate the total number of people in Group B.
        # Since every department must belong to exactly one group (A or B),
        # the sum for Group B is the total sum minus the sum for Group A.
        current_sum_B = total_sum - current_sum_A
        
        # Find the maximum number of people in either group for the current partition.
        # This is the value we want to minimize across all partitions.
        current_max = max(current_sum_A, current_sum_B)
        
        # Update the overall minimum maximum group size found so far.
        # If the maximum size for the current partition (`current_max`) is smaller than 
        # the minimum found previously (`min_max_group_sum`), update `min_max_group_sum`.
        min_max_group_sum = min(min_max_group_sum, current_max)

    # After checking all 2^N possible partitions (subsets for Group A),
    # `min_max_group_sum` will hold the minimum possible value of the maximum group size.
    # Print the final result to standard output.
    print(min_max_group_sum)

# Call the solve function to execute the program logic.
solve()