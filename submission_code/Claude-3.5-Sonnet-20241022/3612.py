class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        def isStrictlyIncreasing(start: int, length: int) -> bool:
            # Check if subarray starting at 'start' of length 'length' is strictly increasing
            for i in range(start, start + length - 1):
                if nums[i] >= nums[i + 1]:
                    return False
            return True
        
        # Check each possible pair of adjacent subarrays
        # The first subarray can start from index 0 to len(nums) - 2*k
        for i in range(len(nums) - 2*k + 1):
            # Check if both subarrays are strictly increasing
            # First subarray starts at i, second starts at i+k
            if isStrictlyIncreasing(i, k) and isStrictlyIncreasing(i + k, k):
                return True
                
        return False