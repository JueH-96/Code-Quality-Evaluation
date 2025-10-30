class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        min_sum = float('inf')
        found = False
        for i in range(len(nums) - l + 1):
            for j in range(l, min(r + 1, len(nums) - i + 1)):
                sub_array = nums[i:i+j]
                current_sum = sum(sub_array)
                if current_sum > 0:
                    min_sum = min(min_sum, current_sum)
                    found = True
        if found:
            return min_sum
        else:
            return -1