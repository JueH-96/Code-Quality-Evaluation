class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        max_value = float('-inf')
        n = len(nums)
        for i in range(n - 2):
            for j in range(i + 1, n -1):
                for k in range(j + 1, n):
                    value = (nums[i] - nums[j]) * nums[k]
                    if value > max_value:
                        max_value = value
        return max(0, max_value)