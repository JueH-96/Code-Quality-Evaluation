import sys

def main():
    """
    This function implements a "jump" simulation to find the final positions of the bars.
    Instead of moving one unit at a time, each bar jumps to its lowest possible position
    in the current configuration of all other bars. This is repeated in rounds until no
    bar can move further. The check for obstacles is done by iterating through all other bars,
    leading to an O(N^2) complexity per round.
    """
    
    # Fast I/O
    # input = sys.stdin.readline

    # Read problem parameters
    try:
        H_str, W_str, N_str = sys.stdin.readline().split()
        H, W, N = int(H_str), int(W_str), int(N_str)
        bars_data = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    except (IOError, ValueError):
        # Handle potential empty input on some platforms
        return

    # R_final stores the current row of each bar. Initialize with starting rows.
    # The index corresponds to the bar's original 1-based index minus one.
    R_final = [b[0] for b in bars_data]

    while True:
        has_changed_in_round = False
        
        # In each round, process bars from 1 to N (0 to N-1 in 0-based index)
        for i in range(N):
            current_r = R_final[i]
            
            # A bar on the bottom row cannot fall further.
            if current_r == H:
                continue

            # Determine the highest obstacle below the current bar.
            # The floor acts as an obstacle at row H, so the bar would stop at H.
            # This means the blocking row is effectively H + 1.
            min_blocking_row = H + 1
            
            # Get column span of the current bar
            c_i_start = bars_data[i][1]
            c_i_end = c_i_start + bars_data[i][2] - 1

            # Check every other bar to see if it's a potential obstacle.
            for k in range(N):
                if i == k:
                    continue
                
                r_k = R_final[k]
                
                # An obstacle must be strictly below the current bar.
                if r_k > current_r:
                    # And their column ranges must overlap.
                    c_k_start = bars_data[k][1]
                    c_k_end = c_k_start + bars_data[k][2] - 1
                    
                    if max(c_i_start, c_k_start) <= min(c_i_end, c_k_end):
                        # This bar is a blocker. Update the highest blocker's row found so far.
                        min_blocking_row = min(min_blocking_row, r_k)
            
            # The bar will fall to the row just above the highest obstacle.
            target_r = min_blocking_row - 1
            
            if target_r > current_r:
                # The bar falls. Update its position and mark that a change occurred.
                R_final[i] = target_r
                has_changed_in_round = True
        
        # If no bar moved in a full round, the system has stabilized.
        if not has_changed_in_round:
            break
            
    # Print the final row for each bar.
    for r in R_final:
        print(r)

if __name__ == "__main__":
    main()