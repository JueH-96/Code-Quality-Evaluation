class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        dp = {}

        def solve(index, prev1, prev2):
            if index == n:
                return 1
            
            if (index, prev1, prev2) in dp:
                return dp[(index, prev1, prev2)]
            
            ans = 0
            for val1 in range(prev1, nums[index] + 1):
                val2 = nums[index] - val1
                if val2 >= 0 and val2 <= prev2:
                    ans = (ans + solve(index + 1, val1, val2)) % mod
            
            dp[(index, prev1, prev2)] = ans
            return ans

        return solve(0, 0, 50)