class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        min_positive_sum = float('inf')
        found = False
        
        # Iterate over all possible starting points of subarrays
        for start in range(n):
            current_sum = 0
            # Iterate over all possible lengths from l to r
            for length in range(l, r + 1):
                end = start + length - 1
                # Ensure the end index is within bounds
                if end >= n:
                    break
                # Add the current element to the current subarray sum
                current_sum += nums[end]
                # Check if the current subarray sum is positive and update the minimum sum found
                if current_sum > 0:
                    found = True
                    min_positive_sum = min(min_positive_sum, current_sum)
        
        # Return the minimum positive sum if found, otherwise return -1
        return min_positive_sum if found else -1