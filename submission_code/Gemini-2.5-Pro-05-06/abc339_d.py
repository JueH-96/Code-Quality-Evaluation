import sys
import collections

def main():
    N = int(sys.stdin.readline())
    S_grid_str = [sys.stdin.readline().strip() for _ in range(N)]

    # grid_char_map stores the grid layout, with 'P' cells converted to '.'
    grid_char_map = [] 
    player_initial_coords = []
    for r_idx in range(N):
        # Create a mutable list of characters for the row
        row_chars = list(S_grid_str[r_idx]) 
        for c_idx in range(N):
            char = row_chars[c_idx]
            if char == 'P':
                player_initial_coords.append((r_idx, c_idx))
                # Treat player cells as empty for movement logic
                row_chars[c_idx] = '.' 
        grid_char_map.append(row_chars)

    p1_r_start, p1_c_start = player_initial_coords[0]
    p2_r_start, p2_c_start = player_initial_coords[1]

    # BFS queue stores tuples: (p1_r, p1_c, p2_r, p2_c, current_dist)
    queue = collections.deque()
    
    # dist_map stores {state_tuple: min_dist}
    # A state is (p1_r, p1_c, p2_r, p2_c)
    dist_map = {}

    initial_state_tuple = (p1_r_start, p1_c_start, p2_r_start, p2_c_start)
    
    # Add initial state to queue and dist_map
    # The problem statement guarantees players start on distinct cells.
    queue.append(initial_state_tuple + (0,)) # Append current_dist = 0
    dist_map[initial_state_tuple] = 0

    min_moves_found = -1

    # Define movement directions: (delta_row, delta_col)
    # Up, Down, Left, Right
    directions = [
        (-1, 0), 
        (1, 0),  
        (0, -1), 
        (0, 1)   
    ]

    while queue:
        p1_r, p1_c, p2_r, p2_c, d = queue.popleft()

        # Target condition: players are on the same cell
        if p1_r == p2_r and p1_c == p2_c:
            min_moves_found = d
            break # BFS guarantees this is the shortest path

        # Explore next states by trying all four directions
        for dr, dc in directions:
            # Calculate player 1's next position
            next_p1_r_candidate, next_p1_c_candidate = p1_r + dr, p1_c + dc
            # Assume player 1 stays initially, update if move is valid
            moved_p1_r, moved_p1_c = p1_r, p1_c 
            if 0 <= next_p1_r_candidate < N and \
               0 <= next_p1_c_candidate < N and \
               grid_char_map[next_p1_r_candidate][next_p1_c_candidate] != '#':
                moved_p1_r, moved_p1_c = next_p1_r_candidate, next_p1_c_candidate
            
            # Calculate player 2's next position
            next_p2_r_candidate, next_p2_c_candidate = p2_r + dr, p2_c + dc
            # Assume player 2 stays initially, update if move is valid
            moved_p2_r, moved_p2_c = p2_r, p2_c 
            if 0 <= next_p2_r_candidate < N and \
               0 <= next_p2_c_candidate < N and \
               grid_char_map[next_p2_r_candidate][next_p2_c_candidate] != '#':
                moved_p2_r, moved_p2_c = next_p2_r_candidate, next_p2_c_candidate

            next_state_tuple = (moved_p1_r, moved_p1_c, moved_p2_r, moved_p2_c)

            # If this state has not been visited yet
            if next_state_tuple not in dist_map:
                dist_map[next_state_tuple] = d + 1
                queue.append(next_state_tuple + (d + 1,))
    
    sys.stdout.write(str(min_moves_found) + "
")

if __name__ == '__main__':
    main()