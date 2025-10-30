import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    S = data[2:]
    
    # Convert each string to a bitmask
    bitmasks = []
    for s in S:
        mask = 0
        for i, char in enumerate(s):
            if char == 'o':
                mask |= (1 << i)
        bitmasks.append(mask)
    
    # We need to cover all flavors, which means we need to cover the bitmask 2^M - 1
    all_flavors = (1 << M) - 1
    
    # Use dynamic programming to find the minimum number of stands needed to cover all flavors
    # dp[mask] will hold the minimum number of stands needed to cover the flavors represented by 'mask'
    dp = [float('inf')] * (1 << M)
    dp[0] = 0  # No stands needed to cover no flavors
    
    # Iterate over all possible sets of stands
    for mask in range(1 << N):
        # Calculate the combined bitmask of this set of stands
        combined_mask = 0
        count = 0
        for i in range(N):
            if mask & (1 << i):
                combined_mask |= bitmasks[i]
                count += 1
        
        # Update dp for the combination of previous dp results and this new combined mask
        for prev_mask in range(1 << M):
            new_mask = prev_mask | combined_mask
            dp[new_mask] = min(dp[new_mask], dp[prev_mask] + count)
    
    # The answer is the minimum number of stands needed to cover all flavors
    print(dp[all_flavors])