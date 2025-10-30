import sys

def solve():
    # Read the input N
    N = int(sys.stdin.readline())
    
    # Read the input string A, split into elements, and convert to a list of integers
    A_str = sys.stdin.readline().split()
    A = [int(x) for x in A_str]

    # Calculate the actual final value by simulating the operations N times
    current_A_list = list(A)
    for _ in range(N):
        next_A_list = []
        # The length of the current list must be a multiple of 3 at each step,
        # as the initial length is 3^N and each step divides the length by 3.
        current_length = len(current_A_list)
        
        # Process the current list in groups of 3
        for i in range(current_length // 3):
            # Get the group of 3 elements
            group = current_A_list[3*i : 3*i+3]
            
            # Count 0s and 1s in the group
            count_0 = group.count(0)
            # count_1 = group.count(1) # Since there are 3 elements, count_1 = 3 - count_0
            count_1 = 3 - count_0

            # Determine the majority value (0 if more 0s, 1 otherwise)
            majority = 0 if count_0 > count_1 else 1
            
            # Append the majority value to the next level string
            next_A_list.append(majority)
        
        # The result of this operation becomes the input for the next operation
        current_A_list = next_A_list
    
    # After N operations, current_A_list contains a single element, which is the final value
    actual_final_value = current_A_list[0]

    # Dynamic Programming approach to find the minimum cost
    # dp[k][i] will store a list [cost_to_make_0, cost_to_make_1] for the i-th bit at level k.
    # Level 0 represents the original string A. Level N represents the final single bit.
    # The size of the string at level k is 3^(N-k).

    # Initialize DP table structure. dp is a list where dp[k] will store costs for level k.
    dp = [None] * (N + 1)

    # Base case: Level 0 (the original string A)
    # Size at level 0 is 3^N
    size_level_0 = 3**N
    dp[0] = [[0, 0] for _ in range(size_level_0)]
    
    # The cost to make an original bit A[j] equal to 0 is 0 if it's already 0, else 1.
    # The cost to make it 1 is 0 if it's already 1, else 1.
    for j in range(size_level_0):
        dp[0][j] = [0 if A[j] == 0 else 1, 0 if A[j] == 1 else 1]

    # Compute DP for levels from 1 up to N (bottom-up)
    for k in range(1, N + 1):
        # The size of the string (number of bits) at the current level k is 3^(N-k)
        size_current_level = 3**(N - k)
        
        # Initialize the DP list for the current level with infinite costs.
        # We will find the minimum cost to achieve a target value, so initialize with a very large value.
        dp[k] = [[float('inf'), float('inf')] for _ in range(size_current_level)]

        # Iterate through each bit at the current level (index i)
        for i in range(size_current_level):
            # The i-th bit at level k is determined by the majority of three bits at level k-1.
            # These three children bits are located at indices 3*i, 3*i+1, and 3*i+2 in the list representation of level k-1.
            child_indices = [3 * i, 3 * i + 1, 3 * i + 2]

            # Retrieve the pre-computed minimum costs for the children bits from the previous level (k-1)
            costs_children = [
                dp[k-1][child_indices[0]], # Costs for child 0 (at index 3*i in level k-1)
                dp[k-1][child_indices[1]], # Costs for child 1 (at index 3*i+1 in level k-1)
                dp[k-1][child_indices[2]]  # Costs for child 2 (at index 3*i+2 in level k-1)
            ]

            # Calculate the minimum cost to make the current bit (at index i, level k) equal to 0.
            # The current bit is 0 if the majority of its three children is 0.
            # The possible combinations of child values (v1, v2, v3) that result in a majority of 0 are:
            # (0,0,0), (0,0,1), (0,1,0), (1,0,0).
            cost_to_make_0 = float('inf')
            # Calculate the total cost for each combination of child values and take the minimum:
            # Combination (0,0,0): sum of costs to make child 0 as 0, child 1 as 0, child 2 as 0
            cost_to_make_0 = min(cost_to_make_0,
                                 costs_children[0][0] + costs_children[1][0] + costs_children[2][0])
            # Combination (0,0,1): sum of costs to make child 0 as 0, child 1 as 0, child 2 as 1
            cost_to_make_0 = min(cost_to_make_0,
                                 costs_children[0][0] + costs_children[1][0] + costs_children[2][1])
            # Combination (0,1,0): sum of costs to make child 0 as 0, child 1 as 1, child 2 as 0
            cost_to_make_0 = min(cost_to_make_0,
                                 costs_children[0][0] + costs_children[1][1] + costs_children[2][0])
            # Combination (1,0,0): sum of costs to make child 0 as 1, child 1 as 0, child 2 as 0
            cost_to_make_0 = min(cost_to_make_0,
                                 costs_children[0][1] + costs_children[1][0] + costs_children[2][0])
            
            # Store the minimum cost to make the current bit 0
            dp[k][i][0] = cost_to_make_0

            # Calculate the minimum cost to make the current bit (at index i, level k) equal to 1.
            # The current bit is 1 if the majority of its three children is 1.
            # The possible combinations of child values (v1, v2, v3) that result in a majority of 1 are:
            # (0,1,1), (1,0,1), (1,1,0), (1,1,1).
            cost_to_make_1 = float('inf')
            # Calculate the total cost for each combination of child values and take the minimum:
            # Combination (0,1,1): sum of costs to make child 0 as 0, child 1 as 1, child 2 as 1
            cost_to_make_1 = min(cost_to_make_1,
                                 costs_children[0][0] + costs_children[1][1] + costs_children[2][1])
            # Combination (1,0,1): sum of costs to make child 0 as 1, child 1 as 0, child 2 as 1
            cost_to_make_1 = min(cost_to_make_1,
                                 costs_children[0][1] + costs_children[1][0] + costs_children[2][1])
            # Combination (1,1,0): sum of costs to make child 0 as 1, child 1 as 1, child 2 as 0
            cost_to_make_1 = min(cost_to_make_1,
                                 costs_children[0][1] + costs_children[1][1] + costs_children[2][0])
            # Combination (1,1,1): sum of costs to make child 0 as 1, child 1 as 1, child 2 as 1
            cost_to_make_1 = min(cost_to_make_1,
                                 costs_children[0][1] + costs_children[1][1] + costs_children[2][1])

            # Store the minimum cost to make the current bit 1
            dp[k][i][1] = cost_to_make_1

        # After completing level k, the DP results for level k-1 are implicitly handled
        # by Python's memory management as dp[k-1] is no longer needed for subsequent levels.

    # The final result is the single bit at level N, which is at index 0.
    # We want to change this final value from its actual value to the opposite value.
    # If actual_final_value is 0, the target is 1. If actual_final_value is 1, the target is 0.
    target_final_value = 1 - actual_final_value
    
    # The minimum cost required to achieve the target final value is stored in the DP table
    # at level N, index 0, for the target value.
    min_cost_to_change = dp[N][0][target_final_value]

    # Print the calculated minimum cost to standard output
    sys.stdout.write(str(min_cost_to_change) + '
')

# Execute the solve function to run the program
solve()