class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = {}

        def solve(index, k1, k2):
            if index == n:
                if k1 == 0 and k2 == 0:
                    return 0
                else:
                    return -1

            if (index, k1, k2) in dp:
                return dp[(index, k1, k2)]

            res = -1
            if k1 > 0:
                res = max(res, solve(index + 1, k1 - 1, k2) | nums[index])
            if k2 > 0:
                res = max(res, solve(index + 1, k1, k2 - 1) ^ nums[index])
            
            res = max(res, solve(index + 1, k1, k2))

            dp[(index, k1, k2)] = res
            return res

        return solve(0, k, k)