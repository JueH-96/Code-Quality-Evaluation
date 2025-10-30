import sys
import collections

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # Preprocess input to build adjacency lists
    # element_to_sets[e]: list of set_indices (0 to N-1) that contain element 'e' (1 to M)
    element_to_sets = [[] for _ in range(M + 1)]
    # set_to_elements[s_idx]: list of element_ids (1 to M) contained in set 's_idx' (0 to N-1)
    set_to_elements = [[] for _ in range(N)]

    for i in range(N):
        A_i = int(sys.stdin.readline())
        elements_in_set = list(map(int, sys.stdin.readline().split()))
        set_to_elements[i] = elements_in_set
        for element in elements_in_set:
            element_to_sets[element].append(i)

    # dist_elements[e]: minimum operations to make element 'e' reachable from '1'
    dist_elements = [float('inf')] * (M + 1)
    # dist_sets[s_idx]: minimum operations to make set 's_idx' reachable from '1'
    dist_sets = [float('inf')] * N

    q = collections.deque()

    # Initialization:
    # 1. Check for the 0-operation case (1 and M are in the same initial set).
    # 2. Add initial sets (containing element 1) to the queue with 0 cost.
    
    found_initial_solution_zero_ops = False
    for s_idx in element_to_sets[1]:
        # If set 's_idx' contains both 1 and M, then 0 operations are needed.
        if M in set_to_elements[s_idx]:
            found_initial_solution_zero_ops = True
            break
        
        # If this set has not been processed yet (cost is infinity),
        # mark it as reachable with 0 operations and add to the deque.
        if dist_sets[s_idx] == float('inf'):
            dist_sets[s_idx] = 0
            # Add to the front of deque as it's a 0-cost initial activation
            q.appendleft((s_idx, 'set')) 

    if found_initial_solution_zero_ops:
        print(0)
        return

    # Main 0-1 BFS loop
    while q:
        node, node_type = q.popleft() # Pop from front (0-cost preferred)

        if node_type == 'set':
            # We are at a set node; all elements within this set can be reached with same cost.
            current_cost = dist_sets[node]
            for element_e in set_to_elements[node]:
                # If we found a cheaper path to this element
                if current_cost < dist_elements[element_e]:
                    dist_elements[element_e] = current_cost
                    # Add element to the front of deque (0-cost transition from set to its elements)
                    q.appendleft((element_e, 'element'))
        
        elif node_type == 'element':
            # We are at an element node; we can use it to activate other sets.
            current_cost = dist_elements[node]
            for set_s_idx in element_to_sets[node]:
                # If using this element to activate set_s_idx yields a cheaper path
                # (cost increases by 1 for a new set merge operation)
                if current_cost + 1 < dist_sets[set_s_idx]:
                    dist_sets[set_s_idx] = current_cost + 1
                    # Add set to the back of deque (1-cost transition from element to new set)
                    q.append((set_s_idx, 'set'))

    # After BFS, dist_elements[M] holds the minimum operations.
    result = dist_elements[M]
    if result == float('inf'):
        print(-1) # M is not reachable from 1
    else:
        print(result)

# Call the solver function
solve()