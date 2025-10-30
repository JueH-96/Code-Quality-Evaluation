class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        min_sum = float('inf')
        
        # Iterate over all possible starting points of subarrays
        for start in range(n):
            current_sum = 0
            # Iterate over all possible lengths from l to r
            for end in range(start, min(start + r, n)):
                current_sum += nums[end]
                # Check if the current subarray length is within the desired range
                if end - start + 1 >= l:
                    # Check if the current sum is greater than 0 and update min_sum
                    if current_sum > 0:
                        min_sum = min(min_sum, current_sum)
        
        # If min_sum was updated, return it, otherwise return -1
        return min_sum if min_sum != float('inf') else -1