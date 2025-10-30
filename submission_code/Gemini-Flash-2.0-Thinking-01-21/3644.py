class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        min_sum = float('inf')
        found_positive_sum = False
        n = len(nums)
        for i in range(n):
            for length in range(l, r + 1):
                if i + length <= n:
                    subarray = nums[i:i+length]
                    current_sum = sum(subarray)
                    if current_sum > 0:
                        min_sum = min(min_sum, current_sum)
                        found_positive_sum = True
        if found_positive_sum:
            return min_sum
        else:
            return -1