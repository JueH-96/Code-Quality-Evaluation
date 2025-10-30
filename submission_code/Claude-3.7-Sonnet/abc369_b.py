def min_fatigue():
    n = int(input().strip())
    
    A = []
    S = []
    for i in range(n):
        a, s = input().strip().split()
        A.append(int(a))
        S.append(s)
    
    # dp[l][r] = minimum fatigue after playing the current set of notes, 
    # with the left hand at position l and the right hand at position r.
    dp = [[0] * 101 for _ in range(101)]  # Initial fatigue is 0
    
    for i in range(n):
        key = A[i]
        hand = S[i]
        
        # Create a new dp array for this iteration
        new_dp = [[float('inf')] * 101 for _ in range(101)]
        
        if hand == 'L':  # Using left hand
            for r in range(1, 101):  # Fixed right hand position
                min_fatigue = float('inf')
                for l in range(1, 101):  # All possible left hand positions
                    min_fatigue = min(min_fatigue, dp[l][r] + abs(key - l))
                new_dp[key][r] = min_fatigue
        else:  # hand == 'R', using right hand
            for l in range(1, 101):  # Fixed left hand position
                min_fatigue = float('inf')
                for r in range(1, 101):  # All possible right hand positions
                    min_fatigue = min(min_fatigue, dp[l][r] + abs(key - r))
                new_dp[l][key] = min_fatigue
        
        dp = new_dp
    
    # Find the minimum fatigue among all hand positions
    min_fatigue = float('inf')
    for l in range(1, 101):
        for r in range(1, 101):
            min_fatigue = min(min_fatigue, dp[l][r])
    
    return min_fatigue

print(min_fatigue())