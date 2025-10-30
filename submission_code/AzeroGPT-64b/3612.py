class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)

        # Function to check if a subarray is strictly increasing
        def is_increasing(subarray):
            return all(subarray[i] < subarray[i + 1] for i in range(len(subarray) - 1))

        # Check all possible subarrays of length k
        for i in range(n - k + 1):
            if is_increasing(nums[i:i+k]) and i + k < n and is_increasing(nums[i+k:i+k+k]):
                return True
        return False