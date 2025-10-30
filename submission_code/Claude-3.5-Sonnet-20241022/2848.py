class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[mask][last] represents number of valid permutations using numbers in mask, ending with last
        @cache
        def dfs(mask, last):
            # If all numbers are used
            if mask == 0:
                return 1
                
            ans = 0
            # Try each unused number as next in permutation
            for i in range(n):
                if mask & (1 << i):  # if number is available
                    curr = nums[i]
                    # Check divisibility condition
                    if last == -1 or curr % last == 0 or last % curr == 0:
                        # Remove current number from mask and recurse
                        ans = (ans + dfs(mask ^ (1 << i), curr)) % MOD
            
            return ans
        
        # Start with all numbers available (mask has all 1s) and no last number (-1)
        return dfs((1 << n) - 1, -1)