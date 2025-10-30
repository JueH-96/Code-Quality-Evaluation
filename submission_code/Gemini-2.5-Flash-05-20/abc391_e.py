import sys

def solve():
    N = int(sys.stdin.readline())
    A_str = sys.stdin.readline().strip()
    A = [int(char) for char in A_str]

    # 1. Calculate initial A'_1 by simulating the operation N times
    current_level_values = list(A) # Make a copy for simulation

    for _ in range(N): # Apply the operation N times
        next_level_values = []
        num_elements_in_current_level = len(current_level_values)
        num_groups = num_elements_in_current_level // 3

        for i in range(num_groups):
            b0 = current_level_values[3*i]
            b1 = current_level_values[3*i+1]
            b2 = current_level_values[3*i+2]
            
            # Majority rule: 1 if sum is 2 or 3 (two or three 1s), 0 otherwise
            majority_val = 1 if (b0 + b1 + b2) >= 2 else 0
            next_level_values.append(majority_val)
        current_level_values = next_level_values
    
    initial_A_prime_1 = current_level_values[0]

    # 2. DP Calculation
    # dp_current_level will store [cost_to_make_0, cost_to_make_1] for elements at the current level of aggregation
    
    # Base case: Level N (leaves - the original A elements)
    dp_current_level = []
    for val in A:
        # Cost to make val 0: 0 if val is 0, 1 if val is 1
        # Cost to make val 1: 1 if val is 0, 0 if val is 1
        dp_current_level.append([0 if val == 0 else 1, 0 if val == 1 else 1])

    # Iterate from level N-1 down to 0 (root)
    # The length of dp_current_level effectively represents 3^current_N_value.
    # We are reducing the string length by a factor of 3 in each step.
    for _ in range(N): 
        dp_next_level = []
        num_nodes_at_this_level = len(dp_current_level) // 3

        for node_idx in range(num_nodes_at_this_level):
            child_idx_base = 3 * node_idx
            c0_opts = dp_current_level[child_idx_base]
            c1_opts = dp_current_level[child_idx_base + 1]
            c2_opts = dp_current_level[child_idx_base + 2]

            # Calculate cost to make current node 0 (majority 0)
            # Combinations that result in majority 0: (0,0,0), (0,0,1), (0,1,0), (1,0,0)
            cost_to_make_0 = min(
                c0_opts[0] + c1_opts[0] + c2_opts[0], # Child 0=0, Child 1=0, Child 2=0
                c0_opts[0] + c1_opts[0] + c2_opts[1], # Child 0=0, Child 1=0, Child 2=1
                c0_opts[0] + c1_opts[1] + c2_opts[0], # Child 0=0, Child 1=1, Child 2=0
                c0_opts[1] + c1_opts[0] + c2_opts[0]  # Child 0=1, Child 1=0, Child 2=0
            )

            # Calculate cost to make current node 1 (majority 1)
            # Combinations that result in majority 1: (1,1,1), (1,1,0), (1,0,1), (0,1,1)
            cost_to_make_1 = min(
                c0_opts[1] + c1_opts[1] + c2_opts[1], # Child 0=1, Child 1=1, Child 2=1
                c0_opts[1] + c1_opts[1] + c2_opts[0], # Child 0=1, Child 1=1, Child 2=0
                c0_opts[1] + c1_opts[0] + c2_opts[1], # Child 0=1, Child 1=0, Child 2=1
                c0_opts[0] + c1_opts[1] + c2_opts[1]  # Child 0=0, Child 1=1, Child 2=1
            )
            dp_next_level.append([cost_to_make_0, cost_to_make_1])
        dp_current_level = dp_next_level

    # The final answer is the cost to change A'_1 from its initial value to the opposite.
    target_value = 1 - initial_A_prime_1
    print(dp_current_level[0][target_value])

solve()