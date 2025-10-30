class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_diff = float('inf')
        
        # Iterate over all possible starting points of subarrays
        for start in range(n):
            current_or = 0
            # Iterate over all possible ending points of subarrays starting from 'start'
            for end in range(start, n):
                # Calculate the OR for the current subarray
                current_or |= nums[end]
                # Calculate the absolute difference with k
                current_diff = abs(k - current_or)
                # Update the minimum difference found so far
                min_diff = min(min_diff, current_diff)
                
                # Early exit if we find a perfect match
                if min_diff == 0:
                    return 0
        
        return min_diff