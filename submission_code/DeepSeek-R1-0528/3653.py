class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_rem = [None] * k
        min_rem[0] = 0
        prefix = 0
        ans = -10**18
        
        for j in range(1, n + 1):
            prefix += nums[j - 1]
            r = j % k
            if min_rem[r] is not None:
                candidate = prefix - min_rem[r]
                if candidate > ans:
                    ans = candidate
            if min_rem[r] is None or prefix < min_rem[r]:
                min_rem[r] = prefix
        
        return ans