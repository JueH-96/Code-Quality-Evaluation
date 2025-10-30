class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        # Sort requirements by end index
        requirements.sort()
        
        # Create mapping of end index to required count
        req_map = {end: cnt for end, cnt in requirements}
        
        # dp[used][inv] represents number of ways to create permutation with 
        # given used mask and current inversion count
        dp = {}
        
        def count_inversions(perm, end):
            inv = 0
            for i in range(end + 1):
                for j in range(i):
                    if perm[j] > perm[i]:
                        inv += 1
            return inv
        
        def solve(used, curr_perm):
            if len(curr_perm) == n:
                return 1
            
            curr_pos = len(curr_perm)
            state = (used, tuple(curr_perm))
            
            if state in dp:
                return dp[state]
            
            total = 0
            # Try placing each unused number at current position
            for num in range(n):
                if used & (1 << num):
                    continue
                    
                # Place the number
                curr_perm.append(num)
                
                # Check if this position needs to satisfy requirement
                valid = True
                if curr_pos in req_map:
                    if count_inversions(curr_perm, curr_pos) != req_map[curr_pos]:
                        valid = False
                
                if valid:
                    total = (total + solve(used | (1 << num), curr_perm)) % MOD
                    
                curr_perm.pop()
                
            dp[state] = total
            return total
        
        return solve(0, [])