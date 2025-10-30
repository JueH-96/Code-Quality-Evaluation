def compute_min_t():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    T_prev = 0
    
    for h in H:
        s = T_prev + 1
        r = (-s) % 3
        
        low = 1
        high = h
        ans = high  # Initialize with maximum possible
        
        while low <= high:
            mid = (low + high) // 2
            full = mid // 3
            rem = mid % 3
            
            cnt = full
            if r < rem:
                cnt += 1
            
            sum_dmg = mid + 2 * cnt
            
            if sum_dmg >= h:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        T_prev += ans
    
    print(T_prev)

compute_min_t()