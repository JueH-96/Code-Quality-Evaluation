class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        if n == 0:
            return 0
        
        from collections import defaultdict
        
        dp = defaultdict(int)
        a = s[0]
        
        # Initialize DP for the first round (i=0)
        for c in ['F', 'W', 'E']:
            # Compute delta based on a and c
            if (a == 'F' and c == 'E') or (a == 'W' and c == 'F') or (a == 'E' and c == 'W'):
                delta = 1
            elif (a == 'F' and c == 'W') or (a == 'W' and c == 'E') or (a == 'E' and c == 'F'):
                delta = -1
            else:
                delta = 0
            
            key = (c, delta)
            dp[key] = (dp[key] + 1) % MOD
        
        for i in range(1, n):
            next_dp = defaultdict(int)
            a = s[i]
            for (last_c, current_diff), count in dp.items():
                for c in ['F', 'W', 'E']:
                    if c == last_c:
                        continue  # Consecutive same creatures are not allowed
                    
                    # Compute delta for this transition
                    if (a == 'F' and c == 'E') or (a == 'W' and c == 'F') or (a == 'E' and c == 'W'):
                        delta = 1
                    elif (a == 'F' and c == 'W') or (a == 'W' and c == 'E') or (a == 'E' and c == 'F'):
                        delta = -1
                    else:
                        delta = 0
                    
                    new_diff = current_diff + delta
                    key = (c, new_diff)
                    next_dp[key] = (next_dp[key] + count) % MOD
            
            dp = next_dp
        
        # Sum all states where Bob's points are strictly greater than Alice's
        total = 0
        for (last_c, diff), count in dp.items():
            if diff > 0:
                total = (total + count) % MOD
        
        return total % MOD