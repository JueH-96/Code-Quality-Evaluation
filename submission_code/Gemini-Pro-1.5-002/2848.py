class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        
        dp = {}
        
        def solve(current_perm):
            if len(current_perm) == n:
                return 1
            
            state = tuple(current_perm)
            if state in dp:
                return dp[state]
            
            count = 0
            for num in nums:
                if num not in current_perm:
                    if not current_perm or (current_perm[-1] % num == 0 or num % current_perm[-1] == 0):
                        count = (count + solve(current_perm + [num])) % MOD
                        
            dp[state] = count
            return count
        
        return solve([])