import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    if S == T:
        print(0)
        return

    mapping = {}
    for i in range(N):
        s_char = S[i]
        t_char = T[i]
        if s_char not in mapping:
            mapping[s_char] = t_char
        elif mapping[s_char] != t_char:
            print(-1)
            return
    
    node_state = [0] * 26  # 0: UNVISITED, 1: VISITING, 2: VISITED
    num_ops = 0

    for i in range(26):
        char_code_start_dfs = i
        char_val_start_dfs = chr(ord('a') + char_code_start_dfs)

        if char_val_start_dfs not in mapping:
            node_state[char_code_start_dfs] = 2 # Not in S, effectively visited
            continue
        if mapping[char_val_start_dfs] == char_val_start_dfs:
            node_state[char_code_start_dfs] = 2 # Maps to self, effectively visited
            continue
        
        if node_state[char_code_start_dfs] == 2: # Already processed
            continue
        
        # If node_state is 1 (VISITING), it implies we've re-entered part of a cycle already
        # being explored. This should not happen if we correctly mark nodes VISITED.
        # So, we must be starting from an UNVISITED node.
        if node_state[char_code_start_dfs] == 1: # Should not happen
            continue


        current_dfs_path_codes = []
        
        curr_char_val = char_val_start_dfs
        curr_char_code = char_code_start_dfs
        
        while node_state[curr_char_code] == 0: # UNVISITED
            node_state[curr_char_code] = 1 # Mark VISITING
            current_dfs_path_codes.append(curr_char_code)
            
            next_char_val = mapping[curr_char_val]
            next_char_code = ord(next_char_val) - ord('a')

            if next_char_val not in mapping or mapping[next_char_val] == next_char_val: # Sink
                num_ops += len(current_dfs_path_codes)
                for path_node_code in current_dfs_path_codes:
                    node_state[path_node_code] = 2 # Mark VISITED
                
                if node_state[next_char_code] == 0: # Mark sink as VISITED too if it wasn't
                     node_state[next_char_code] = 2
                
                current_dfs_path_codes = [] # Path processed
                break 
            
            curr_char_val = next_char_val
            curr_char_code = next_char_code

        if not current_dfs_path_codes: # Path ended in a sink and was processed
            continue

        # Loop terminated because node_state[curr_char_code] is not UNVISITED
        if node_state[curr_char_code] == 1: # VISITING: Cycle detected
            num_ops += 1 # Cycle overhead
            num_ops += len(current_dfs_path_codes) # Edges in the component
            for path_node_code in current_dfs_path_codes:
                node_state[path_node_code] = 2 # Mark VISITED
        
        elif node_state[curr_char_code] == 2: # VISITED: Path leads to already processed component
            num_ops += len(current_dfs_path_codes) # Edges in this new path segment
            for path_node_code in current_dfs_path_codes:
                node_state[path_node_code] = 2 # Mark VISITED
                
    print(num_ops)

solve()