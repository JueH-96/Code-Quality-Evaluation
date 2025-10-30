class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        non_decreasing_count = 0
        
        # Helper function to calculate the number of operations needed to make a subarray non-decreasing
        def operations_needed(subarray):
            ops = 0
            for i in range(1, len(subarray)):
                if subarray[i] < subarray[i - 1]:
                    ops += subarray[i - 1] - subarray[i]
            return ops
        
        # Check each subarray
        for start in range(n):
            for end in range(start, n):
                subarray = nums[start:end + 1]
                if operations_needed(subarray) <= k:
                    non_decreasing_count += 1
        
        return non_decreasing_count