import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    C = list(map(int, sys.stdin.readline().split()))

    # Identify indices based on (A_i, B_i) and store their costs C_i
    # I_10: Indices where A_i=1 and B_i=0. Need 1 -> 0 flip.
    # I_01: Indices where A_i=0 and B_i=1. Need 0 -> 1 flip.
    # I_11: Indices where A_i=1 and B_i=1. Start at 1, need to end at 1 (net even flips).
    # I_00: Indices where A_i=0 and B_i=0. Start at 0, need to end at 0 (net even flips).

    I_10_costs = [] 
    I_01_costs = []
    I_11_costs = [] 
    
    initial_S = 0

    for i in range(N):
        if A[i] == 1 and B[i] == 0:
            I_10_costs.append(C[i])
            initial_S += C[i]
        elif A[i] == 0 and B[i] == 1:
            I_01_costs.append(C[i])
        elif A[i] == 1 and B[i] == 1:
            I_11_costs.append(C[i])
            initial_S += C[i]
        # A_i=0, B_i=0 indices do not contribute to the initial sum S

    # Strategy:
    # Indices where A_i != B_i must be flipped a net odd number of times.
    # Indices where A_i = B_i must be flipped a net even number of times.
    # Minimal flips for A_i != B_i is 1. Minimal flips for A_i = B_i is 0.
    # We only consider flipping indices where A_i != B_i once.
    # Indices with A_i=1 contribute C_i to the current sum S.
    # Flipping 1->0 on index i costs S - C_i and reduces S by C_i.
    # Flipping 0->1 on index i costs S + C_i and increases S by C_i.
    # To minimize total cost, we want to perform operations when S is low.
    # Operations that reduce S (1->0 flips) are preferable early.
    # Operations that increase S (0->1 flips) are preferably late.
    # Among 1->0 flips (indices in I_10), performing on larger C_i first reduces S more aggressively.
    # Among 0->1 flips (indices in I_01), performing on smaller C_i first increases S less aggressively.
    # Indices in I_11 and I_00 are never flipped as it would only add cost (a pair of flips results in a net cost increase).

    # Sort costs for 1->0 flips in descending order
    I_10_costs.sort(reverse=True)
    # Sort costs for 0->1 flips in ascending order
    I_01_costs.sort()

    current_S = initial_S
    total_cost = 0

    # Perform 1->0 flips for indices initially in I_10
    # These indices are currently 1 and need to become 0.
    # Flipping 1->0 reduces the current sum S.
    # The cost is calculated using the sum *after* the flip.
    for c_i in I_10_costs:
        # The current sum S includes C_i because A_i is currently 1
        # After flipping A_i to 0, the new sum S' = S - C_i
        # The cost of this operation is S' = S - C_i
        cost = current_S - c_i
        total_cost += cost
        # Update S for subsequent operations
        current_S -= c_i

    # Perform 0->1 flips for indices initially in I_01
    # These indices are currently 0 and need to become 1.
    # Flipping 0->1 increases the current sum S.
    # The cost is calculated using the sum *after* the flip.
    for c_i in I_01_costs:
        # The current sum S does NOT include C_i because A_i is currently 0
        # After flipping A_i to 1, the new sum S' = S + C_i
        # The cost of this operation is S' = S + C_i
        cost = current_S + c_i
        total_cost += cost
        # Update S for subsequent operations
        current_S += c_i

    # The final state A is identical to B. The process stops.
    # The total cost accumulated is the minimum.

    print(total_cost)

solve()