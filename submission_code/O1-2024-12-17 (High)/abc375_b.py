def main():
    import sys
    import math
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    
    # Start at origin
    cur_x, cur_y = 0, 0
    idx = 1
    total_dist = 0.0
    
    # Visit each point in order
    for _ in range(N):
        x, y = int(data[idx]), int(data[idx+1])
        total_dist += math.hypot(x - cur_x, y - cur_y)
        cur_x, cur_y = x, y
        idx += 2
        
    # Return to origin
    total_dist += math.hypot(cur_x, cur_y)
    
    # Print the result with sufficient precision
    print(f"{total_dist:.10f}")

# Do not forget to call main()
if __name__ == "__main__":
    main()