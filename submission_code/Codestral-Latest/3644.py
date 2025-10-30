class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        min_sum = float('inf')
        found = False

        for length in range(l, r + 1):
            for i in range(len(nums) - length + 1):
                subarray_sum = sum(nums[i:i + length])
                if subarray_sum > 0:
                    min_sum = min(min_sum, subarray_sum)
                    found = True

        return min_sum if found else -1