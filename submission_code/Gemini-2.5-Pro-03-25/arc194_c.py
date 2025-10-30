# YOUR CODE HERE
import heapq
import sys

def solve():
    # Read the size of the sequences
    N = int(sys.stdin.readline())
    
    # Read sequence A
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read sequence B
    B = list(map(int, sys.stdin.readline().split()))
    
    # Read sequence C (costs)
    C = list(map(int, sys.stdin.readline().split()))

    # Identify the indices where A[i] differs from B[i]. These are the indices that must be flipped an odd number of times.
    # To minimize cost, we assume each differing index needs to be flipped exactly once.
    diff_indices = []
    for i in range(N):
        if A[i] != B[i]:
            diff_indices.append(i)

    # If A is already identical to B, no operations are needed, and the cost is 0.
    if not diff_indices:
        print(0)
        return

    # We maintain the current state of sequence A. Make a mutable copy.
    current_A = list(A) 
    
    # Calculate the initial value of the sum S = sum(A_k * C_k for k where A_k = 1).
    # This sum represents the cost calculation basis.
    current_S = 0
    for i in range(N):
        if current_A[i] == 1:
            current_S += C[i]

    # We use a min-priority queue (min-heap) to efficiently find the greedy choice at each step.
    # The priority queue stores tuples (P_i, index).
    # P_i represents the change in the sum S if we flip A[i].
    # If A[i] is 0, flipping it to 1 increases the sum by C[i], so P_i = C[i].
    # If A[i] is 1, flipping it to 0 decreases the sum by C[i], so P_i = -C[i].
    # The greedy strategy is to choose the flip i that minimizes the resulting state's sum, which is current_S + P_i.
    # Minimizing current_S + P_i is equivalent to minimizing P_i for a fixed current_S.
    pq = [] 

    # Populate the priority queue based on the initial state A and the differing indices.
    # The proof of correctness for the greedy strategy relies on P_i values computed based on the state *before* the flip.
    # Since the state A_i for an index i that hasn't been flipped yet remains unchanged from the initial state,
    # computing P_i based on the initial state A is correct.
    for i in diff_indices:
        P_i = 0
        if current_A[i] == 0: # Flip 0 -> 1. Change in sum = +C[i].
            P_i = C[i]
        else: # current_A[i] == 1. Flip 1 -> 0. Change in sum = -C[i].
            P_i = -C[i]
        
        # Push the calculated P_i value and the corresponding index onto the min-heap.
        heapq.heappush(pq, (P_i, i))

    # Initialize the total cost accumulator.
    total_cost = 0
    
    # The number of flips required is the number of indices where A and B initially differ.
    num_flips_needed = len(diff_indices)

    # Perform the flips one by one, following the greedy strategy.
    # The loop runs exactly K times, where K is the number of necessary flips.
    for _ in range(num_flips_needed):
        
        # Failsafe check: if the priority queue becomes empty prematurely, something is wrong.
        # This should not happen if K = len(diff_indices) and each diff_index was pushed once.
        if not pq:
             break 
             
        # Extract the element with the smallest P_i value from the priority queue.
        # This corresponds to the greedy choice: the flip that results in the minimum cost state.
        # P_i_star is the change in the sum S caused by flipping A[i_star].
        P_i_star, i_star = heapq.heappop(pq)
        
        # The cost incurred for performing this operation is defined as the sum S of the state *after* the flip.
        # This value equals the sum before the flip (current_S) plus the change incurred by the flip (P_i_star).
        cost_this_step = current_S + P_i_star
        
        # Add the cost of this step to the total accumulated cost.
        total_cost += cost_this_step
        
        # Update the current sum S to reflect the sum of the new state after the flip.
        # The sum of the new state *is* precisely cost_this_step.
        current_S = cost_this_step 

        # Optionally update the tracked state array current_A. Not strictly necessary for the algorithm's logic
        # because P_i values depend only on the initial state for un-flipped indices.
        # current_A[i_star] = 1 - current_A[i_star] # Flip the bit at index i_star
        
    # After performing all necessary flips, print the minimum total cost found.
    print(total_cost)

# Execute the main logic of the problem solution.
solve()