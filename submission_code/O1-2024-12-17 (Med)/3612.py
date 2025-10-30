class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        # Helper function to check if a subarray of length k starting at 'start'
        # is strictly increasing.
        def is_strictly_increasing(arr, start, length):
            for i in range(start, start + length - 1):
                if arr[i] >= arr[i + 1]:
                    return False
            return True
        
        n = len(nums)
        # We only need to check subarrays starting up to index n - 2*k
        # because we need space for two subarrays of length k that are adjacent.
        for i in range(n - 2*k + 1):
            # Check if subarray [i..i+k-1] is strictly increasing
            # and the immediately following subarray [i+k..i+2k-1] is also strictly increasing.
            if is_strictly_increasing(nums, i, k) and is_strictly_increasing(nums, i + k, k):
                return True
        
        return False