def solve():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    W = int(input_data[0])
    H = int(input_data[1])
    N = int(input_data[2])
    idx = 3
    strawberries = []
    for _ in range(N):
        p = int(input_data[idx]); q = int(input_data[idx+1])
        idx += 2
        strawberries.append((p, q))
    A = int(input_data[idx]); idx += 1
    cut_x = list(map(int, input_data[idx:idx+A]))
    idx += A
    B = int(input_data[idx]); idx += 1
    cut_y = list(map(int, input_data[idx:idx+B]))
    idx += B
    
    # Prepare the boundary arrays (including 0 and W for x, 0 and H for y)
    X = [0] + cut_x + [W]
    Y = [0] + cut_y + [H]
    
    # A dictionary (or defaultdict) to count strawberries in each cell
    from collections import defaultdict
    cell_count = defaultdict(int)
    
    # For each strawberry, find which cell it belongs to
    # iv = index in X intervals, ih = index in Y intervals
    for (px, py) in strawberries:
        iv = bisect.bisect_left(X, px) - 1
        ih = bisect.bisect_left(Y, py) - 1
        cell_count[(iv, ih)] += 1
    
    total_cells = (A + 1) * (B + 1)
    used_cells = len(cell_count)
    
    # If not all cells are used by strawberries, min is 0
    # Otherwise, min is the smallest cell count
    if used_cells < total_cells:
        min_val = 0
    else:
        min_val = min(cell_count.values())
    
    max_val = max(cell_count.values()) if cell_count else 0  # cell_count won't be empty if N>0
    
    print(min_val, max_val)

def main():
    solve()

if __name__ == "__main__":
    main()