def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W, K = map(int, input_data[:3])
    grid = input_data[3:]
    
    # Edge case: If K == 1, we just need a single 'o' or '.' anywhere.
    #   - If there's an 'o', cost 0
    #   - Else if there's a '.', cost 1
    #   - Else -1
    if K == 1:
        if any('o' in row for row in grid):
            print(0)
            return
        elif any('.' in row for row in grid):
            print(1)
            return
        else:
            print(-1)
            return
    
    # We'll find the minimal cost among:
    # 1) Horizontal segments of length K that have no 'x'
    # 2) Vertical segments of length K that have no 'x'
    # Approach:
    #   For each row, break into segments separated by 'x'.
    #   For each segment of length >= K, use a sliding window to find
    #       how many '.' appear in each sub-window of length K.
    #   Track the minimum number of '.' across all such sub-windows.
    # We do the same for columns.
    # If we never find a valid sub-window, answer is -1.

    INF = float('inf')
    min_cost = INF
    
    # Helper: compute min '.' in any length-K sub-window for a given list of cells
    #   where 2 = 'x', 1='.', 0='o'
    def min_dot_in_segment(arr, K):
        # arr does NOT contain 'x' (2). We just do a sliding window of size K,
        # counting how many 1's in each window. We can do prefix sums.
        n = len(arr)
        if n < K:
            return INF
        # Build prefix sums:
        #   p[i] = number of '.' from arr[0] to arr[i-1]
        p = [0]*(n+1)
        for i in range(n):
            p[i+1] = p[i] + (1 if arr[i] == 1 else 0)
        # Now the cost of sub-window [i, i+K) is p[i+K] - p[i]
        res = INF
        for i in range(n-K+1):
            cost = p[i+K] - p[i]
            if cost < res:
                res = cost
        return res
    
    # Convert each row into numeric form: 0='o', 1='.', 2='x'
    numeric_grid = []
    for row in grid:
        arr = []
        for ch in row:
            if ch == 'o':
                arr.append(0)
            elif ch == '.':
                arr.append(1)
            else:  # x
                arr.append(2)
        numeric_grid.append(arr)
    
    # Check horizontal segments
    for i in range(H):
        row_data = numeric_grid[i]
        start = 0
        while start < W:
            if row_data[start] == 2:
                # skip 'x'
                start += 1
                continue
            # gather a segment of non-'x'
            end = start
            while end < W and row_data[end] != 2:
                end += 1
            segment = row_data[start:end]
            
            cost_segment = min_dot_in_segment(segment, K)
            if cost_segment < min_cost:
                min_cost = cost_segment
            
            start = end  # move on
    
    # Check vertical segments
    # We'll iterate column by column
    # We'll build the column as a list, then do the same segmentation logic
    # Because H*W <= 2e5, this is feasible
    for j in range(W):
        start = 0
        while start < H:
            if numeric_grid[start][j] == 2:
                start += 1
                continue
            end = start
            while end < H and numeric_grid[end][j] != 2:
                end += 1
            segment = [numeric_grid[r][j] for r in range(start, end)]
            
            cost_segment = min_dot_in_segment(segment, K)
            if cost_segment < min_cost:
                min_cost = cost_segment
            
            start = end
    
    # If min_cost is INF, it is impossible
    if min_cost == INF:
        print(-1)
    else:
        print(min_cost)

# Do not forget to call main()!
if __name__ == "__main__":
    main()