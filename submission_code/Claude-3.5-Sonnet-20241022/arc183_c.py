def solve():
    MOD = 998244353
    N, M = map(int, input().split())
    conditions = []
    for _ in range(M):
        l, r, x = map(int, input().split())
        conditions.append((l-1, r-1, x-1))  # Convert to 0-based indexing
        
    # dp[S][p] = number of valid permutations using numbers in set S 
    # where p positions are filled from left
    dp = {}
    
    def check_conditions(perm, length):
        # Check if current partial permutation violates any condition
        for l, r, x in conditions:
            if r < length:
                # If condition range is fully within current length
                if x <= r and max(perm[l:r+1]) == perm[x]:
                    return False
            elif l < length:
                # If condition range partially overlaps
                if x < length and max(perm[l:length]) == perm[x]:
                    return False
        return True
    
    def solve_dp(used, pos):
        if pos == N:
            return 1
            
        state = (used, pos)
        if state in dp:
            return dp[state]
            
        ans = 0
        used_set = set()
        mask = used
        while mask:
            used_set.add(mask & -mask)
            mask &= mask - 1
            
        for i in range(N):
            if (1 << i) & used:
                continue
                
            # Try placing number i+1 at position pos
            new_perm = []
            temp = used
            for p in range(pos):
                lowest_bit = temp & -temp
                new_perm.append(bin(lowest_bit).count('0') - 1)
                temp &= temp - 1
            new_perm.append(i)
            
            if check_conditions(new_perm, pos + 1):
                ans = (ans + solve_dp(used | (1 << i), pos + 1)) % MOD
                
        dp[state] = ans
        return ans
        
    result = solve_dp(0, 0)
    print(result)

solve()