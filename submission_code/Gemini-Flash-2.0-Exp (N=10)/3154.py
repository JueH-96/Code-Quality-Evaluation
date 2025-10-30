class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_val = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    val = (nums[i] - nums[j]) * nums[k]
                    max_val = max(max_val, val)
        return max_val