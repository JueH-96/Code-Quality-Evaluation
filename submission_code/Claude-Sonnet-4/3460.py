class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        # Create requirements map
        req = {}
        for end, cnt in requirements:
            req[end] = cnt
        
        # dp[i][j] = number of permutations of first i+1 positions with j inversions
        from functools import lru_cache
        
        @lru_cache(None)
        def solve(pos, inversions):
            if pos == n:
                return 1
            
            # If this position has a requirement, check it
            if pos in req and inversions != req[pos]:
                return 0
            
            result = 0
            # When we place an element at position pos, it can create 0 to pos inversions
            # with the elements already placed
            for k in range(pos + 1):
                new_inversions = inversions + k
                # Pruning: check if it's possible to satisfy future requirements
                remaining_positions = n - pos - 1
                max_future_inversions = remaining_positions * (remaining_positions + 1) // 2
                
                valid = True
                for future_pos in req:
                    if future_pos > pos:
                        if new_inversions > req[future_pos] or new_inversions + max_future_inversions < req[future_pos]:
                            valid = False
                            break
                
                if valid:
                    result = (result + solve(pos + 1, new_inversions)) % MOD
            
            return result
        
        return solve(0, 0)