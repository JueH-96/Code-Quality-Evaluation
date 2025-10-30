class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # We have two adjacent subarrays, each of length k.
        # The starting index a of the first subarray must be such that a + 2*k - 1 is within bounds.
        for a in range(n - 2 * k + 1):
            # Check if the first subarray [a, a+k-1] is strictly increasing.
            first_strict = all(nums[i] < nums[i + 1] for i in range(a, a + k - 1))
            # Check if the second subarray [a+k, a+2*k-1] is strictly increasing.
            second_strict = all(nums[i] < nums[i + 1] for i in range(a + k, a + 2 * k - 1))
            
            if first_strict and second_strict:
                return True
        return False