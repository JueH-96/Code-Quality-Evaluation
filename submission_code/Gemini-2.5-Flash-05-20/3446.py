from typing import List

class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        """
        Calculates the total number of "good pairs" (i, j).
        A pair (i, j) is good if nums1[i] is divisible by nums2[j] * k.

        Args:
            nums1: An integer array of length n.
            nums2: An integer array of length m.
            k: A positive integer.

        Returns:
            The total count of good pairs.
        """
        
        # Initialize a counter for good pairs.
        good_pairs_count = 0
        
        # Get the lengths of the input arrays.
        n = len(nums1)
        m = len(nums2)
        
        # Iterate through each element in nums1 using its index i.
        for i in range(n):
            # For each element nums1[i], iterate through each element in nums2
            # using its index j.
            for j in range(m):
                # Calculate the potential divisor based on the problem definition: nums2[j] * k.
                # According to the constraints (1 <= nums2[j], 1 <= k),
                # the divisor will always be a positive integer (>= 1),
                # so there's no risk of division by zero.
                divisor = nums2[j] * k
                
                # Check if nums1[i] is perfectly divisible by the calculated divisor.
                # If the remainder of the division is 0, it means nums1[i] is divisible by 'divisor'.
                if nums1[i] % divisor == 0:
                    # If it is divisible, increment the count of good pairs.
                    good_pairs_count += 1
        
        # After checking all possible pairs (i, j), return the total count.
        return good_pairs_count