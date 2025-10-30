class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)

        # Helper function to check if a subarray is strictly increasing
        def is_increasing(subarray):
            for i in range(1, len(subarray)):
                if subarray[i] <= subarray[i - 1]:
                    return False
            return True

        # Iterate through the array to find adjacent increasing subarrays of length k
        for i in range(n - k - k + 1):
            subarray1 = nums[i:i + k]
            subarray2 = nums[i + k:i + 2 * k]

            if is_increasing(subarray1) and is_increasing(subarray2):
                return True

        return False