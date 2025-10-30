class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        for i in range(n):
            max_val = nums[i]
            min_val = nums[i]
            for j in range(i, n):
                max_val = max(max_val, nums[j])
                min_val = min(min_val, nums[j])
                if j - i + 1 <= 2 and max_val - min_val <= 2:
                    res += 1
        return res