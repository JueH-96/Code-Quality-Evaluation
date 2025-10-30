# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W, N = map(int, sys.stdin.readline().split())
    T = sys.stdin.readline().strip()
    S = [sys.stdin.readline().strip() for _ in range(H)]
    H_grid = H
    W_grid = W
    N_moves = N
    T_moves = T
    Grid = S

    delta_i = [0] * (N_moves + 1)
    delta_j = [0] * (N_moves + 1)
    min_i = [0] * (N_moves + 1)
    max_i = [0] * (N_moves + 1)
    min_j = [0] * (N_moves + 1)
    max_j = [0] * (N_moves + 1)

    for t in range(1, N_moves +1):
        move = T_moves[t-1]
        delta_i[t] = delta_i[t-1]
        delta_j[t] = delta_j[t-1]
        if move == 'U':
            delta_i[t] -=1
        elif move == 'D':
            delta_i[t] +=1
        elif move == 'L':
            delta_j[t] -=1
        elif move == 'R':
            delta_j[t] +=1

        min_i[t] = min(min_i[t-1], delta_i[t])
        max_i[t] = max(max_i[t-1], delta_i[t])
        min_j[t] = min(min_j[t-1], delta_j[t])
        max_j[t] = max(max_j[t-1], delta_j[t])

    I_start_min = 1 - min_i[N_moves]
    I_start_max = H_grid - max_i[N_moves]
    J_start_min = 1 - min_j[N_moves]
    J_start_max = W_grid - max_j[N_moves]

    # Adjust ranges to be within 1..H and 1..W
    I_start_min = max(1, I_start_min)
    I_start_max = min(H_grid, I_start_max)
    J_start_min = max(1, J_start_min)
    J_start_max = min(W_grid, J_start_max)

    # Initialize mask of valid starting positions
    mask = [[True]*(W_grid+2) for _ in range(H_grid+2)]

    # Initialize all positions outside the valid starting ranges as False
    for i0 in range(0, H_grid+2):
        for j0 in range(0, W_grid+2):
            if not (I_start_min <= i0 <= I_start_max and J_start_min <= j0 <= J_start_max):
                mask[i0][j0] = False

    # Collect sea cells
    sea_cells = []
    for i in range(1, H_grid +1):
        for j in range(1, W_grid +1):
            if Grid[i-1][j-1] == '#':
                sea_cells.append((i, j))

    # For each sea cell and each t, mark invalid starting positions
    for t in range(N_moves +1):
        shift_i = delta_i[t]
        shift_j = delta_j[t]
        for (i_s, j_s) in sea_cells:
            i0 = i_s - shift_i
            j0 = j_s - shift_j
            if I_start_min <= i0 <= I_start_max and J_start_min <= j0 <= J_start_max:
                # Ensure shifted positions are within grid
                if 1 <= i0 <= H_grid and 1 <= j0 <= W_grid:
                    mask[i0][j0] = False

    # Collect valid starting positions
    ending_positions = set()
    for i0 in range(I_start_min, I_start_max +1):
        for j0 in range(J_start_min, J_start_max +1):
            if not mask[i0][j0]:
                continue
            # Check that starting position is land
            if Grid[i0-1][j0-1] == '#':
                continue
            valid = True
            for t in range(N_moves +1):
                i_t = i0 + delta_i[t]
                j_t = j0 + delta_j[t]
                if not (1 <= i_t <= H_grid and 1 <= j_t <= W_grid):
                    valid = False
                    break
                if Grid[i_t-1][j_t-1] == '#':
                    valid = False
                    break
            if valid:
                i_end = i0 + delta_i[N_moves]
                j_end = j0 + delta_j[N_moves]
                ending_positions.add((i_end, j_end))

    print(len(ending_positions))


threading.Thread(target=main).start()