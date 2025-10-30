class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = {}

        def countSubsequences(index, currentSum):
            if index == n:
                if currentSum == k:
                    return 1
                else:
                    return 0
            
            if (index, currentSum) in dp:
                return dp[(index, currentSum)]

            include = countSubsequences(index + 1, currentSum + nums[index])
            exclude = countSubsequences(index + 1, currentSum)
            
            dp[(index, currentSum)] = (include + exclude) % MOD
            return dp[(index, currentSum)]

        return countSubsequences(0, 0)