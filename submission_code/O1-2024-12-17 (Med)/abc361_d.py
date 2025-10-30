def main():
    import sys
    from collections import deque
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = input_data[1]
    T = input_data[2]
    
    # Quick check: the number of black/white stones must match
    if S.count('B') != T.count('B') or S.count('W') != T.count('W'):
        print(-1)
        return
    
    # If already the same, answer is 0 (no operations needed)
    if S == T:
        print(0)
        return
    
    # Each state is (e, stone_string), where:
    #   e in [0..N], meaning the empty cells are at positions e and e+1 (0-based) among N+2 cells.
    #   stone_string is the left-to-right concatenation of the N stones (B/W) skipping the empty cells.
    #
    # Initially, e = N because cells N+1 and N+2 (1-based) are empty => in 0-based that's positions N, N+1.
    # We want to reach a state (e= N, stone_string=T).
    
    # Build the initial state.
    start_e = N  # empties at the far right
    start_stones = S
    
    # Target state.
    goal_e = N
    goal_stones = T
    
    # A function to build the full arrangement of length N+2, given (e, stones).
    # e is 0-based index of the left empty cell. We place '.' in positions e, e+1 of a 0-based array of length N+2.
    # Then we fill the other N positions from left to right using 'stones'.
    def build_arrangement(e, stones_str):
        arr = ['.'] * (N+2)
        idx = 0
        for i in range(N+2):
            if i == e or i == e+1:
                arr[i] = '.'
            else:
                arr[i] = stones_str[idx]
                idx += 1
        return arr
    
    # Precompute a key for visited dictionary:
    # We'll just use a simple string "e:stones" as the key.
    def state_key(e, stones_str):
        return str(e) + ":" + stones_str
    
    # BFS
    start_key = state_key(start_e, start_stones)
    queue = deque()
    queue.append((start_e, start_stones, 0))  # (e, stones, dist)
    visited = set()
    visited.add(start_key)
    
    while queue:
        e, stones_str, dist = queue.popleft()
        # Build the full arrangement for the current state
        arr = build_arrangement(e, stones_str)
        
        # Try moving each adjacent pair of stones (x, x+1) into the empty pair (e, e+1)
        for x in range(N+1):
            if arr[x] != '.' and arr[x+1] != '.':
                # We can swap the pair at x, x+1 with the empties e, e+1
                new_arr = arr[:]
                # Move arr[x], arr[x+1] => new_arr[e], new_arr[e+1]
                new_arr[e] = arr[x]
                new_arr[e+1] = arr[x+1]
                # Now x, x+1 become empty
                new_arr[x] = '.'
                new_arr[x+1] = '.'
                
                e2 = x  # new empty pair is at x, x+1
                # Build the new stones string by concatenating non-empty cells
                new_stones = ''.join(c for c in new_arr if c != '.')
                
                new_key = state_key(e2, new_stones)
                if new_key not in visited:
                    visited.add(new_key)
                    new_dist = dist + 1
                    # Check if this is the goal
                    if e2 == goal_e and new_stones == goal_stones:
                        print(new_dist)
                        return
                    queue.append((e2, new_stones, new_dist))
    
    # If we exhaust all possibilities without finding the goal, it's impossible
    print(-1)

# Don't forget to call main()
if __name__ == "__main__":
    main()