class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # The maximum number of subarrays is achieved by splitting at every zero.
        # This is because the AND of any number with zero is zero, which is the minimum possible score.
        # Therefore, we just need to count the number of zeros in the array.
        return nums.count(0) + 1  # We add 1 because even if there are no zeros, we have at least one subarray.