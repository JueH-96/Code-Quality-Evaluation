import math

# Helper function to calculate 3^N
def power_of_three(n):
    return int(math.pow(3, n))

# Function to compute the majority of a triplet (as integers 0 or 1)
def majority(a, b, c):
    return 1 if a + b + c >= 2 else 0

def solve():
    N = int(input())
    # Read the binary string as space-separated integers
    A = list(map(int, input().split()))

    size = power_of_three(N)

    # Calculate the original final value by applying the majority operation N times
    current_A = list(A) # Copy the initial string
    # We calculate level 1 string, then level 2, ..., up to level N string (which has 1 element)
    for level in range(N): # level 0 -> level 1, level 1 -> level 2, ..., level N-1 -> level N
        next_A = []
        # Number of nodes/groups at the next level ('level + 1')
        num_groups = power_of_three(N - (level + 1))
        for i in range(num_groups):
            # Indices in the current level ('level') array
            idx0 = 3 * i
            idx1 = 3 * i + 1
            idx2 = 3 * i + 2
            next_A.append(majority(current_A[idx0], current_A[idx1], current_A[idx2]))
        current_A = next_A # current_A now holds values for level `level + 1`

    original_final_value = current_A[0]

    # Dynamic programming to find minimum costs
    # costs[i][0] = min cost to make node i at current level have value 0
    # costs[i][1] = min cost to make node i at current level have value 1
    # Start with costs at level 0 (leaf nodes)
    # `costs` will store costs for the level we are currently computing FROM.
    # Initially, `costs` holds level 0 costs.
    costs = [[0, 0] for _ in range(size)] # size 3^N x 2

    for i in range(size):
        if A[i] == 0:
            costs[i][0] = 0 # Cost to make 0 is 0 if it's already 0
            costs[i][1] = 1 # Cost to make 1 is 1 if it's 0
        else: # A[i] == 1
            costs[i][0] = 1 # Cost to make 0 is 1 if it's 1
            costs[i][1] = 0 # Cost to make 1 is 0 if it's already 1

    # Iterate N times to compute costs up to level N
    for level in range(N): # Compute costs for level 1, then level 2, ..., up to level N
        # We compute costs for the next level (level 'level + 1') based on costs from the current level ('level')
        num_nodes_curr = power_of_three(N - (level + 1)) # Number of nodes at the next level
        next_costs = [[0, 0] for _ in range(num_nodes_curr)]

        # Iterate through each node at the next level
        for i in range(num_nodes_curr):
            # Children indices in the current level ('level') costs array
            idx0 = 3 * i
            idx1 = 3 * i + 1
            idx2 = 3 * i + 2

            # Costs for the three children nodes (at level 'level')
            c00, c01 = costs[idx0] # Child 0 costs
            c10, c11 = costs[idx1] # Child 1 costs
            c20, c21 = costs[idx2] # Child 2 costs

            # Calculate costs for the current node (at level 'level+1', index i) to be 0 (majority 0)
            # Need children values to sum <= 1
            # Combinations (0,0,0), (1,0,0), (0,1,0), (0,0,1)
            s000 = c00 + c10 + c20 # Target children values (0,0,0)
            s100 = c01 + c10 + c20 # Target children values (1,0,0)
            s010 = c00 + c11 + c20 # Target children values (0,1,0)
            s001 = c00 + c10 + c21 # Target children values (0,0,1)
            next_costs[i][0] = min(s000, s100, s010, s001)

            # Calculate costs for the current node (at level 'level+1', index i) to be 1 (majority 1)
            # Need children values to sum >= 2
            # Combinations (1,1,0), (1,0,1), (0,1,1), (1,1,1)
            s110 = c01 + c11 + c20 # Target children values (1,1,0)
            s101 = c01 + c10 + c21 # Target children values (1,0,1)
            s011 = c00 + c11 + c21 # Target children values (0,1,1)
            s111 = c01 + c11 + c21 # Target children values (1,1,1)
            next_costs[i][1] = min(s110, s101, s011, s111)

        costs = next_costs # `costs` now holds costs for the level just computed (`level + 1`)

    # After N iterations, `costs` has size 1x2, storing costs for the root node (level N)
    # costs[0][0] is min cost to make final value 0
    # costs[0][1] is min cost to make final value 1

    # Determine the minimum changes needed to flip the original final value
    if original_final_value == 0:
        # Original was 0, we need to change it to 1. The minimum cost is stored in costs[0][1].
        print(costs[0][1])
    else: # original_final_value == 1
        # Original was 1, we need to change it to 0. The minimum cost is stored in costs[0][0].
        print(costs[0][0])

solve()