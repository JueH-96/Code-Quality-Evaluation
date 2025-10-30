class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        min_sum = None  # To store the minimum positive sum found
        
        # Check all subarray lengths from l to r
        for length in range(l, r + 1):
            # Check each subarray of the current length
            for start in range(n - length + 1):
                current_sum = sum(nums[start:start+length])
                if current_sum > 0:
                    if min_sum is None or current_sum < min_sum:
                        min_sum = current_sum
        
        return min_sum if min_sum is not None else -1