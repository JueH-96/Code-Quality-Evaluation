import collections
import sys

def main():
    N = int(sys.stdin.readline())
    S_str = sys.stdin.readline().strip()
    T_str = sys.stdin.readline().strip()

    # Check if transformation is possible based on stone counts
    if S_str.count('W') != T_str.count('W'):
        print("-1")
        return

    # Initial state setup
    # config_tuple represents all N+2 cells. Indices 0 to N-1 are for stones, N and N+1 are empty.
    initial_config_list = list(S_str)
    initial_config_list.append('.')
    initial_config_list.append('.')
    initial_config_tuple = tuple(initial_config_list)
    initial_empty_idx = N  # 0-indexed: cells N and N+1 are empty
    initial_state = (initial_config_tuple, initial_empty_idx)

    # Target state setup
    target_config_list = list(T_str)
    target_config_list.append('.')
    target_config_list.append('.')
    target_config_tuple = tuple(target_config_list)
    target_empty_idx = N # Target also has empty cells at N, N+1
    
    # BFS queue stores ((config_tuple, empty_idx), distance)
    q = collections.deque()
    # dist stores {state: min_distance}
    dist = {} 

    q.append((initial_state, 0))
    dist[initial_state] = 0
    
    while q:
        current_state_pair, d = q.popleft()
        current_config_tuple, current_empty_idx = current_state_pair
        
        # Check if target reached
        if current_config_tuple == target_config_tuple and current_empty_idx == target_empty_idx:
            print(d)
            return

        # k_0 is the 0-indexed start of the current pair of empty cells
        k_0 = current_empty_idx 
        
        # Iterate over possible pairs of cells (x_0, x_0+1) to pick stones from
        # x_0 is 0-indexed, ranges from 0 to N. (So x_0+1 ranges from 1 to N+1)
        # (N+2) cells are indexed 0 to N+1. A pair (x_0, x_0+1) implies x_0 can be at most N.
        for x_0 in range(N + 1): 
            # Check if cells x_0 and x_0+1 both contain stones
            if current_config_tuple[x_0] == '.' or current_config_tuple[x_0+1] == '.':
                continue
            
            # Create new configuration from current one
            new_config_list = list(current_config_tuple)

            # Stones to be moved
            stone_from_x_0 = new_config_list[x_0]
            stone_from_x_0_plus_1 = new_config_list[x_0+1]

            # Move stones to current empty cells (k_0, k_0+1)
            new_config_list[k_0] = stone_from_x_0
            new_config_list[k_0+1] = stone_from_x_0_plus_1
            
            # Cells (x_0, x_0+1) from where stones were taken become empty
            new_config_list[x_0] = '.'
            new_config_list[x_0+1] = '.'
            
            final_new_config_tuple = tuple(new_config_list) 
            new_empty_idx = x_0 # The new empty cells are at x_0, x_0+1
            
            next_state = (final_new_config_tuple, new_empty_idx)

            # If this new state hasn't been visited, add to queue and dist map
            if next_state not in dist: 
                dist[next_state] = d + 1
                q.append((next_state, d + 1))
                
    # If queue becomes empty and target not reached
    print("-1")

if __name__ == '__main__':
    main()