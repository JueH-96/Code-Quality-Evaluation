class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        # Function to check if a subarray of length k is strictly increasing
        def is_strictly_increasing(start):
            for i in range(start, start + k - 1):
                if nums[i] >= nums[i + 1]:
                    return False
            return True
        
        # Iterate over possible starting points for the first subarray
        for a in range(n - 2 * k + 1):
            if is_strictly_increasing(a) and is_strictly_increasing(a + k):
                return True
        
        return False