def maximumTotalCost(nums):
    n = len(nums)
    if n == 0:
        return 0
    
    # Compute the prefix sums with alternating signs
    S = [0] * (n + 1)
    for i in range(1, n + 1):
        S[i] = S[i-1] + nums[i-1] * ((-1) ** (i-1))
    
    dp = [0] * n
    max_a1 = -float('inf')
    max_a2 = -float('inf')
    
    # Process j = -1 (virtual start)
    a_j = 0
    c_j = (-1) ** (-1)  # -1
    if c_j == 1:
        if a_j > max_a1:
            max_a1 = a_j
    else:
        if a_j > max_a2:
            max_a2 = a_j
    
    for i in range(n):
        x = S[i+1]
        current_max = -float('inf')
        
        if max_a1 != -float('inf'):
            current_max = max(current_max, max_a1 + x)
        if max_a2 != -float('inf'):
            current_max = max(current_max, max_a2 - x)
        
        dp[i] = current_max
        
        a_i = dp[i] + ((-1) ** i) * x
        c_i = (-1) ** i
        
        if c_i == 1:
            if a_i > max_a1:
                max_a1 = a_i
        else:
            if a_i > max_a2:
                max_a2 = a_i
    
    return max(dp)