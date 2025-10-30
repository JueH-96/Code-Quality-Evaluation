# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math

    sys.setrecursionlimit(1 << 25)
    H, W, N = map(int, sys.stdin.readline().split())
    bars = []
    for _ in range(N):
        R_i, C_i, L_i = map(int, sys.stdin.readline().split())
        bars.append({'R_i': R_i, 'C_i': C_i, 'L_i': L_i})

    # Initialize the grid cells with the bar occupying them
    # For each cell, we keep track of which bar is occupying it
    # Since W and H are up to 2e5, we cannot create a grid of size H x W
    # Instead, we can use a dict or set to keep track of occupied cells

    # cell_occupancy[(row, col)] = bar_index
    cell_occupancy = {}
    for idx, bar in enumerate(bars):
        R_i = bar['R_i']
        C_i = bar['C_i']
        L_i = bar['L_i']
        for j in range(L_i):
            c = C_i + j
            cell_occupancy[(R_i, c)] = idx

    # Now simulate the movement until no bar moves
    moving_bars = set(range(N))
    last_moved = set()
    while True:
        moved = set()
        for i in sorted(moving_bars):
            bar = bars[i]
            R_i = bar['R_i']
            if R_i >= H:
                continue  # Already at bottom
            can_move = True
            C_i, L_i = bar['C_i'], bar['L_i']
            for j in range(L_i):
                c = C_i + j
                r = R_i + 1
                if (r, c) in cell_occupancy:
                    can_move = False
                    break
            if can_move:
                moved.add(i)
        if not moved:
            break
        # Update positions of moved bars
        for i in moved:
            bar = bars[i]
            # Remove old positions
            R_i, C_i, L_i = bar['R_i'], bar['C_i'], bar['L_i']
            for j in range(L_i):
                c = C_i + j
                del cell_occupancy[(R_i, c)]
            # Move down by 1
            R_i += 1
            bar['R_i'] = R_i
            # Update positions
            for j in range(L_i):
                c = C_i + j
                cell_occupancy[(R_i, c)] = i
        moving_bars = moved

    # Finally, output the R'_i
    for i in range(N):
        print(bars[i]['R_i'])

threading.Thread(target=main).start()