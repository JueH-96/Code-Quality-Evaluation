def solve():
    N, Q = map(int, input().split())
    heights = list(map(int, input().split()))
    
    for _ in range(Q):
        l, r = map(int, input().split())
        l -= 1  # Adjusting to 0-indexed
        r -= 1  # Adjusting to 0-indexed
        
        # Compute max_l_to_r once per query
        max_l_to_r = -1
        for k in range(l+1, r+1):
            max_l_to_r = max(max_l_to_r, heights[k])
        
        count = 0
        for j in range(r+1, N):
            # Check condition 1: Building j can be seen from l
            if heights[j] <= max_l_to_r:
                continue
            
            # Check condition 2: Building j can be seen from r
            max_r_to_j = -1
            for k in range(r+1, j):
                max_r_to_j = max(max_r_to_j, heights[k])
            
            if heights[j] > max_r_to_j:
                count += 1
        
        print(count)

solve()