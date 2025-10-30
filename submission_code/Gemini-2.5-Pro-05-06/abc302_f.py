import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())

    # adj_set_to_int[i] lists elements in set S_i (0-indexed)
    adj_set_to_int = [[] for _ in range(N)]
    # adj_int_to_set[x] lists set indices (0 to N-1) that contain integer x
    adj_int_to_set = [[] for _ in range(M + 1)]

    for i in range(N):
        # A_i is the count of elements, not directly used after reading elements list
        _ = int(sys.stdin.readline()) 
        elements = list(map(int, sys.stdin.readline().split()))
        adj_set_to_int[i] = elements
        for x_val in elements:
            adj_int_to_set[x_val].append(i)
    
    dist_int = [float('inf')] * (M + 1)
    dist_set = [float('inf')] * N
    
    dist_int[1] = 0 
    
    dq = deque()
    # Store (id, item_type) in deque. 
    # item_type 0 for integer, 1 for set_idx.
    dq.append((1, 0)) # (integer 1, type 0)

    while dq:
        curr_id, item_type = dq.popleft()

        if item_type == 0: # Current node is an integer value x
            x = curr_id
            current_dist_for_x = dist_int[x]
            
            for set_idx in adj_int_to_set[x]:
                if current_dist_for_x < dist_set[set_idx]:
                    dist_set[set_idx] = current_dist_for_x
                    dq.appendleft((set_idx, 1))
        
        else: # Current node is a set_idx
            set_idx = curr_id
            current_dist_for_set = dist_set[set_idx]
            
            cost_to_reach_elements_via_this_set = current_dist_for_set + 1

            for val_in_set in adj_set_to_int[set_idx]:
                if cost_to_reach_elements_via_this_set < dist_int[val_in_set]:
                    dist_int[val_in_set] = cost_to_reach_elements_via_this_set
                    dq.append((val_in_set, 0))

    final_chain_length_for_M = dist_int[M]

    if final_chain_length_for_M == float('inf'):
        print("-1")
    else:
        # final_chain_length_for_M is k, the number of sets in the chain.
        # Number of operations is k-1.
        print(final_chain_length_for_M - 1)

if __name__ == '__main__':
    main()