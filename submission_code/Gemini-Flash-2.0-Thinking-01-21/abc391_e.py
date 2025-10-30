# YOUR CODE HERE
import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())
    # Read the binary string A as a list of integers
    # Input format is space-separated integers 0 and 1
    A = list(map(int, sys.stdin.readline().split()))

    # --- Calculate the original final value by applying the majority rule N times ---
    current_A = list(A)
    # Apply the operation N times
    for level in range(N):
        next_A = []
        # The current string length is 3^(N-level). Each group has 3 elements.
        # The next string length is 3^(N-level) / 3 = 3^(N-level-1).
        for i in range(len(current_A) // 3):
            # Get the group of 3 elements for the i-th position in the next level
            group = current_A[3*i : 3*i+3]
            # Calculate the majority value of the group
            majority = 1 if sum(group) >= 2 else 0
            next_A.append(majority)
        # Move to the next level
        current_A = next_A
    # After N operations, current_A has length 1
    original_final_value = current_A[0]

    # --- Dynamic Programming to find minimum costs ---
    # dp[i][0]: min cost (minimum changes in the original string) to make bit i at the current level be 0
    # dp[i][1]: min cost (minimum changes in the original string) to make bit i at the current level be 1
    # Initial level (level 0, t=0) corresponds to the input string A
    
    # Current level size is 3^N initially
    current_level_size = 3**N
    # Initialize the DP table for the base level (level 0)
    dp = [[0, 0] for _ in range(current_level_size)]

    # Base cases for level 0 (the leaves of the computation tree)
    # For each bit A[i] in the original string:
    for i in range(current_level_size):
        # Cost to make A[i] zero: 1 if A[i] is 1 (needs change), 0 otherwise
        dp[i][0] = (A[i] == 1)
        # Cost to make A[i] one: 1 if A[i] is 0 (needs change), 0 otherwise
        dp[i][1] = (A[i] == 0)

    # Iterate through levels, from level 1 up to level N
    # Level t (1 <= t <= N) is computed from level t-1
    # The dp table at the start of iteration t holds costs for level t-1
    for t in range(1, N + 1):
        # Number of elements at level t is 3**(N-t)
        next_level_size = 3**(N-t)
        # Initialize the DP table for the next level (level t)
        next_dp = [[0, 0] for _ in range(next_level_size)]

        # Compute costs for each node at the next level (level t)
        for i in range(next_level_size):
            # The i-th node at level t depends on its three children at level t-1
            # The children are the elements at indices 3*i, 3*i+1, 3*i+2 in the dp table of level t-1
            j0 = 3 * i
            j1 = 3 * i + 1
            j2 = 3 * i + 2

            # Get the costs for the children nodes from the current dp table (level t-1)
            # cX_0: min cost for child X to be 0
            # cX_1: min cost for child X to be 1
            c0_0, c0_1 = dp[j0][0], dp[j0][1]
            c1_0, c1_1 = dp[j1][0], dp[j1][1]
            c2_0, c2_1 = dp[j2][0], dp[j2][1]

            # Calculate min cost to make the current node (at level t, index i) be 0
            # This requires the majority of its 3 children to be 0.
            # The 4 combinations of child values (v0, v1, v2) resulting in majority 0 are:
            # (0,0,0), (0,0,1), (0,1,0), (1,0,0) (and their permutations).
            # We sum the minimum costs for children to achieve these values and take the minimum sum.
            # Cost for (v0, v1, v2) is cost(child0=v0) + cost(child1=v1) + cost(child2=v2).
            next_dp[i][0] = min(
                c0_0 + c1_0 + c2_0, # Children values (0,0,0)
                c0_0 + c1_0 + c2_1, # Children values (0,0,1)
                c0_0 + c1_1 + c2_0, # Children values (0,1,0)
                c0_1 + c1_0 + c2_0  # Children values (1,0,0)
            )

            # Calculate min cost to make the current node (at level t, index i) be 1
            # This requires the majority of its 3 children to be 1.
            # The 4 combinations of child values (v0, v1, v2) resulting in majority 1 are:
            # (1,1,1), (1,1,0), (1,0,1), (0,1,1) (and their permutations).
            # We sum the minimum costs for children to achieve these values and take the minimum sum.
            next_dp[i][1] = min(
                c0_1 + c1_1 + c2_1, # Children values (1,1,1)
                c0_1 + c1_1 + c2_0, # Children values (1,1,0)
                c0_1 + c1_0 + c2_1, # Children values (1,0,1)
                c0_0 + c1_1 + c2_1  # Children values (0,1,1)
            )

        # The computed costs for level t become the current costs for the next iteration (level t+1)
        dp = next_dp

    # After N iterations, dp has size 1x2 and stores the minimum costs for the final bit (at level N)
    # dp[0][0] is the min cost to make the final bit 0
    # dp[0][1] is the min cost to make the final bit 1
    
    # We want to find the minimum changes required to change the original final value.
    # So, the target value is the opposite of the original final value.
    target_final_value = 1 - original_final_value

    # The answer is the minimum cost to achieve the target final value at the root node (index 0)
    print(dp[0][target_final_value])

solve()