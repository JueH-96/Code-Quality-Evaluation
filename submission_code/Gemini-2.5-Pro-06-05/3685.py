class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        """
        Counts the number of subarrays of length 3 where the sum of the first
        and third elements equals half of the second element.

        Args:
            nums: A list of integers.

        Returns:
            The number of such subarrays.
        """
        count = 0
        n = len(nums)

        # The problem constraints state 3 <= nums.length, so we are guaranteed
        # to have at least one subarray of length 3.

        # We need to iterate through all possible subarrays of length 3.
        # A subarray starting at index `i` consists of elements nums[i], nums[i+1],
        # and nums[i+2]. The loop for the starting index `i` will go from 0 to n-3.
        # `range(n - 2)` correctly iterates from i = 0 to n-3.
        for i in range(n - 2):
            # The condition is: nums[i] + nums[i+2] == nums[i+1] / 2
            
            # To avoid floating-point arithmetic and potential precision issues,
            # we can multiply both sides of the equation by 2.
            # The equivalent condition using only integers is:
            # 2 * (nums[i] + nums[i+2]) == nums[i+1]
            if 2 * (nums[i] + nums[i+2]) == nums[i+1]:
                count += 1
        
        return count