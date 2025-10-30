def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    H = int(input_data[0])
    W = int(input_data[1])
    N = int(input_data[2])
    
    holes = set()
    idx = 3
    for _ in range(N):
        r = int(input_data[idx])
        c = int(input_data[idx+1])
        holes.add((r, c))
        idx += 2
    
    # We will use a rolling-array DP approach to save memory:
    # dp_current[j] will represent the size of the largest holeless square
    # ending at row i, column j.
    # dp_previous[j] will be the same but for row i-1.
    
    dp_previous = [0] * (W + 1)
    dp_current = [0] * (W + 1)
    
    total_holeless_squares = 0
    
    for i in range(1, H+1):
        # Swap references and reset dp_current for new row
        dp_previous, dp_current = dp_current, dp_previous
        dp_current[0] = 0  # ensure boundary condition
        for j in range(1, W+1):
            if (i, j) in holes:
                dp_current[j] = 0
            else:
                dp_current[j] = min(dp_previous[j], dp_previous[j-1], dp_current[j-1]) + 1
            total_holeless_squares += dp_current[j]
    
    print(total_holeless_squares)

# Let's call solve() here:
# solve()