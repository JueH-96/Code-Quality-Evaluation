class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        # Helper function to check if nums[start..start+k-1] is strictly increasing
        def is_strictly_increasing(start, length):
            for i in range(start, start + length - 1):
                if nums[i] >= nums[i + 1]:
                    return False
            return True
        
        # We only need to check starting indices up to n - 2k
        # because we need two subarrays of length k that are adjacent.
        for i in range(n - 2*k + 1):
            # Check if nums[i..i+k-1] is strictly increasing
            if is_strictly_increasing(i, k):
                # Check if nums[i+k..i+2k-1] is strictly increasing
                if is_strictly_increasing(i + k, k):
                    return True
        
        return False