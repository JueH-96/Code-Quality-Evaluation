def main():
    import sys
    data = sys.stdin.read().strip().split()
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    idx = 3
    
    # Collect the columns of holed squares for each row in a list
    holes_in_row = [[] for _ in range(H + 1)]
    for _ in range(N):
        r = int(data[idx])
        c = int(data[idx + 1])
        idx += 2
        holes_in_row[r].append(c)
    
    # Sort the hole columns in each row for interval-based processing
    for i in range(1, H + 1):
        holes_in_row[i].sort()
    
    # We only need the previous row (prev) and the current row (cur) in the DP
    prev = [0] * (W + 1)
    cur = [0] * (W + 1)
    
    answer = 0
    
    # DP computation
    # dp(i, j) = 1 + min(dp(i-1, j), dp(i, j-1), dp(i-1, j-1)) if cell(i,j) not holed
    # otherwise dp(i, j) = 0.
    #
    # Here, prev[j]   ≡ dp(i-1, j)
    #       prev[j-1] ≡ dp(i-1, j-1)
    #       cur[j-1]  ≡ dp(i, j-1)
    for i in range(1, H + 1):
        # Clear this row's DP array before filling
        for j in range(W + 1):
            cur[j] = 0
        
        # We'll insert a sentinel hole at W+1 to handle the last interval
        holes = holes_in_row[i] + [W + 1]
        
        start = 1
        for hole_col in holes:
            end = hole_col - 1
            # Fill DP in the holeless interval [start .. end]
            if end >= start:
                for j in range(start, end + 1):
                    cur[j] = 1 + min(prev[j], prev[j - 1], cur[j - 1])
                    answer += cur[j]
            
            # Hole column itself is set to 0 if within the grid
            if hole_col <= W:
                cur[hole_col] = 0
            
            start = hole_col + 1
        
        # Swap the rows
        prev, cur = cur, prev
    
    print(answer)

if __name__ == "__main__":
    main()