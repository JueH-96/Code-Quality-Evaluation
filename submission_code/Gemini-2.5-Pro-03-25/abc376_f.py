# YOUR CODE HERE
import heapq
import sys

# Function to calculate neighbors correctly using 0-based indexing
# state is a tuple (l, r) where l, r are 0-based indices representing the parts held by left and right hands.
# N is the number of parts on the ring.
def get_neighbors(state, N):
    """
    Calculates the valid neighbor states reachable in one operation from the given state.
    An operation consists of moving one hand to an adjacent part, provided the other hand
    is not already on the destination part.
    Returns a list of tuples, each containing (neighbor_state, cost_delta). Cost_delta is always 1.
    """
    l, r = state
    neighbors = []
    
    # Try moving Left Hand
    # Adjacent parts for left hand at position l are (l-1) mod N and (l+1) mod N.
    
    # Move counter-clockwise: new left position is (l - 1 + N) % N
    next_l_ccw = (l - 1 + N) % N
    if next_l_ccw != r: # Check destination is not occupied by right hand
        neighbors.append(((next_l_ccw, r), 1)) # Add neighbor state and cost delta (which is 1)
    
    # Move clockwise: new left position is (l + 1) % N
    next_l_cw = (l + 1) % N
    if next_l_cw != r:
        neighbors.append(((next_l_cw, r), 1))

    # Try moving Right Hand
    # Adjacent parts for right hand at position r are (r-1) mod N and (r+1) mod N.

    # Move counter-clockwise: new right position is (r - 1 + N) % N
    next_r_ccw = (r - 1 + N) % N
    if next_r_ccw != l: # Check destination is not occupied by left hand
         neighbors.append(((l, next_r_ccw), 1))
         
    # Move clockwise: new right position is (r + 1) % N
    next_r_cw = (r + 1) % N
    if next_r_cw != l:
         neighbors.append(((l, next_r_cw), 1))

    return neighbors


def solve():
    N, Q = map(int, sys.stdin.readline().split())
    
    # Use 0-based indexing internally: parts 0, ..., N-1 correspond to parts 1, ..., N
    # Initial state: Left hand on part 1 (index 0), Right hand on part 2 (index 1).
    # current_states maps state tuple (l, r) to the minimum total cost accumulated to reach this state
    # after completing the previous instruction. Initially, only state (0, 1) is reachable with cost 0.
    current_states = {(0, 1): 0} 

    # If there are no instructions (Q=0), no operations needed, total cost is 0.
    if Q == 0:
        print(0)
        return

    # Process each instruction sequentially
    for k in range(Q):
        # Read the k-th instruction (H_i denotes hand, T_i denotes target part number)
        H_i, T_i_str = sys.stdin.readline().split()
        T_i = int(T_i_str) - 1 # Convert target part number to 0-based index

        # Determine the set of target states for this instruction.
        # A target state is any valid state (l, r) where l != r, and the specified hand H_i is on part T_i.
        target_states_check = set()
        if H_i == 'L':
            # Left hand must be at T_i. Right hand can be at any position r_idx except T_i.
            for r_idx in range(N):
                if r_idx != T_i:
                    target_states_check.add((T_i, r_idx))
        else: # H_i == 'R'
            # Right hand must be at T_i. Left hand can be at any position l_idx except T_i.
            for l_idx in range(N):
                if l_idx != T_i:
                    target_states_check.add((l_idx, T_i))

        # Initialize Dijkstra's algorithm structures for finding shortest paths for this instruction.
        pq = [] # Min-heap priority queue: stores tuples (cost, l, r). Ordered by cost.
        
        # dist stores the minimum cost found so far to reach state (l, r) *during this instruction's processing*.
        # It is initialized with the costs from the 'current_states' dictionary, which holds results from the previous step.
        dist = {} 

        # Push initial states (minimum cost states from the previous step) into the priority queue.
        # Their costs are the total costs accumulated up to the previous instruction.
        for state, cost in current_states.items():
             heapq.heappush(pq, (cost, state[0], state[1]))
             dist[state] = cost

        # Dictionary to store the minimum costs found *specifically for the target states* of this instruction.
        final_states_costs = {} 

        # Check if any of the initial states already satisfy the target condition.
        # If so, record their costs in final_states_costs.
        for state, cost in current_states.items():
            if state in target_states_check:
                current_min = final_states_costs.get(state, float('inf'))
                final_states_costs[state] = min(current_min, cost)


        # Main Dijkstra loop to explore states reachable from the initial states.
        while pq:
            # Pop the state with the minimum accumulated cost from the priority queue.
            d, l, r = heapq.heappop(pq)
            current_state = (l, r)
            
            # Optimization: If we've already found a shorter path to this state, skip processing this element.
            if d > dist.get(current_state, float('inf')):
                 continue

            # Check if the current state is one of the target states for this instruction.
            if current_state in target_states_check:
                 # If it's a target state, record or update its minimum cost found so far.
                 current_min_cost_for_this_target = final_states_costs.get(current_state, float('inf'))
                 final_states_costs[current_state] = min(current_min_cost_for_this_target, d)
            
            # Explore neighbors of the current state.
            for neighbor_state, cost_delta in get_neighbors(current_state, N):
                # Calculate the cost to reach the neighbor state through the current state.
                new_dist = d + cost_delta # cost_delta is always 1 for one operation.
                
                # If this path to the neighbor is shorter than any previously known path:
                if new_dist < dist.get(neighbor_state, float('inf')):
                    # Update the minimum distance to the neighbor state.
                    dist[neighbor_state] = new_dist
                    # Push the neighbor state onto the priority queue with its new cost.
                    heapq.heappush(pq, (new_dist, neighbor_state[0], neighbor_state[1]))

        # After Dijkstra finishes, determine the minimum cost achieved among all reachable target states for this instruction.
        min_final_cost = float('inf')
        
        if not final_states_costs:
             # According to the problem statement, any instruction is achievable.
             # This block should ideally not be reached if Q > 0.
             # If it is reached, indicates an error or edge case not handled.
             # Keep min_final_cost as infinity to potentially reveal error downstream.
             # This might occur if N=1 or N=2 but constraint N>=3 prevents this.
             pass # Based on problem constraints and guarantee, assume this is not possible.

        else: # If target states were reached (either initially or via Dijkstra)
             min_final_cost = min(final_states_costs.values())

        # Prepare 'current_states' for the next iteration.
        # It should contain only those target states that achieved the overall minimum cost 'min_final_cost'.
        next_current_states = {}
        for state, cost in final_states_costs.items():
             if cost == min_final_cost:
                 # The cost associated with these states is the total minimum cost up to this instruction.
                 next_current_states[state] = min_final_cost
        
        # Update current_states for the next loop iteration. These states become the starting points for the next instruction's Dijkstra run.
        current_states = next_current_states
        
        # Safety check: If current_states becomes empty, it means no state achieved the min_final_cost.
        # This would imply min_final_cost was infinity, which contradicts problem statement. Check necessary if guarantee might be violated.
        if not current_states and Q > 0:
             # Could indicate an issue. Print debug info or handle error.
             # Assume this won't happen under problem specification.
             pass


    # After processing all Q instructions, the minimum total cost is stored as the value in the current_states dictionary.
    # All states in current_states will have the same cost, which is the minimum total cost.
    if not current_states:
         # This state should only be reachable if Q=0 (handled earlier) or error.
         # Defaulting to 0 or handling based on potential errors.
         # Given problem constraints and guarantee, assume current_states is non-empty if Q > 0.
         # If Q>0 and this happens, it's likely an error state.
         # Let's print 0 as a fail-safe, though ideally error should be raised or identified.
         final_total_cost = 0 

    else:
        # Since all values in current_states are the same minimum cost, we can retrieve any value.
        # Using min() is one way, though direct access like `list(current_states.values())[0]` would also work.
        final_total_cost = min(current_states.values())
    
    print(final_total_cost)

solve()