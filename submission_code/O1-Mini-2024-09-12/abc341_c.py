import sys
import numpy as np

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W, N = map(int, sys.stdin.readline().split())
    T = sys.stdin.readline().strip()
    S = [sys.stdin.readline().strip() for _ in range(H)]
    
    land = np.array([[c == '.' for c in row] for row in S], dtype=bool)
    
    prefix_x = [0] * (N+1)
    prefix_y = [0] * (N+1)
    move_map = {'U': (-1, 0), 'D': (1,0), 'L': (0,-1), 'R': (0,1)}
    
    for k in range(1, N+1):
        dx, dy = move_map[T[k-1]]
        prefix_x[k] = prefix_x[k-1] + dx
        prefix_y[k] = prefix_y[k-1] + dy
    
    total_dx = prefix_x[N]
    total_dy = prefix_y[N]
    
    shifts_x = [prefix_x[k] - total_dx for k in range(N+1)]
    shifts_y = [prefix_y[k] - total_dy for k in range(N+1)]
    
    mask = np.ones((H, W), dtype=bool)
    
    for dx, dy in zip(shifts_x, shifts_y):
        shifted = land.copy()
        # Shift rows
        if dx != 0:
            shifted = np.roll(shifted, -dx, axis=0)
            if dx > 0:
                shifted[-dx:, :] = False
            else:
                shifted[:-dx, :] = False
        # Shift columns
        if dy != 0:
            shifted = np.roll(shifted, -dy, axis=1)
            if dy > 0:
                shifted[:, -dy:] = False
            else:
                shifted[:, :-dy] = False
        # Combine with mask
        mask &= shifted
        if not mask.any():
            break  # Early termination if no possible positions
    
    print(np.sum(mask))

if __name__ == "__main__":
    main()