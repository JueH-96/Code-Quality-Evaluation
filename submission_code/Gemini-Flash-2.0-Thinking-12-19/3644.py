class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        min_sum = float('inf')
        found_subarray = False
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                subarray_len = len(subarray)
                if l <= subarray_len <= r:
                    subarray_sum = sum(subarray)
                    if subarray_sum > 0:
                        min_sum = min(min_sum, subarray_sum)
                        found_subarray = True
        if not found_subarray:
            return -1
        else:
            return min_sum