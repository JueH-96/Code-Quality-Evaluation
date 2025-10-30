class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        
        dp = {}
        
        def solve(index, prev1):
            if index == n:
                return 1
            
            if (index, prev1) in dp:
                return dp[(index, prev1)]
            
            count = 0
            for curr1 in range(prev1, nums[index] + 1):
                curr2 = nums[index] - curr1
                if index > 0:
                    prev2 = nums[index-1] - prev1
                    if curr2 <= prev2:
                        count = (count + solve(index + 1, curr1)) % MOD
                else:
                    count = (count + solve(index + 1, curr1)) % MOD
            
            dp[(index, prev1)] = count
            return count
        
        return solve(0, 0)