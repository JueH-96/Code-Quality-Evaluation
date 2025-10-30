def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W, N = map(int, input_data[:3])
    holes = set()
    idx = 3
    for _ in range(N):
        a, b = map(int, input_data[idx:idx+2])
        idx += 2
        holes.add((a, b))
    
    # We'll keep only two rows of DP to save memory.
    dp_prev = [0] * (W + 1)
    dp_curr = [0] * (W + 1)
    
    answer = 0
    
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            if (i, j) not in holes:
                dp_curr[j] = 1 + min(dp_prev[j], dp_prev[j-1], dp_curr[j-1])
            else:
                dp_curr[j] = 0
            answer += dp_curr[j]
        
        # Swap references so dp_curr becomes dp_prev for the next row,
        # and reset dp_curr to zero for the new row's computation.
        dp_prev, dp_curr = dp_curr, dp_prev
        for k in range(W + 1):
            dp_curr[k] = 0
    
    print(answer)

# Don't forget to call main()
if __name__ == "__main__":
    main()