class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_val = 0

        for j in range(1, n - 1):
            max_i = max(nums[:j])
            max_k = max(nums[j+1:])
            val = (max_i - nums[j]) * max_k
            max_val = max(max_val, val)

        return max_val