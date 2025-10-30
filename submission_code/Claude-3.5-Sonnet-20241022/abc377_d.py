def solve():
    # Read input
    N, M = map(int, input().split())
    intervals = []
    for _ in range(N):
        l, r = map(int, input().split())
        intervals.append((l, r))
    
    # For each possible pair (l,r), check if it's valid
    count = 0
    for l in range(1, M + 1):
        for r in range(l, M + 1):
            valid = True
            # Check each interval [L_i, R_i]
            for L_i, R_i in intervals:
                # If [l,r] completely contains [L_i, R_i], this pair is invalid
                if l <= L_i and R_i <= r:
                    valid = False
                    break
            if valid:
                count += 1
    
    print(count)

# Run the solution
solve()