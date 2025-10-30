import math # Not strictly necessary for the chosen solution, but good practice if considering alternatives
from collections import Counter # Not strictly necessary for the chosen solution
from typing import List # Required for type hinting

class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        """
        Calculates the total number of good pairs (i, j) such that
        nums1[i] is divisible by nums2[j] * k.

        A pair (i, j) is called good if:
        - 0 <= i <= n - 1 (where n is the length of nums1)
        - 0 <= j <= m - 1 (where m is the length of nums2)
        - nums1[i] % (nums2[j] * k) == 0

        Args:
            nums1: The first list of integers (length n).
            nums2: The second list of integers (length m).
            k: The positive integer factor.

        Returns:
            The total count of good pairs.

        Constraints:
            1 <= n, m <= 50
            1 <= nums1[i], nums2[j] <= 50
            1 <= k <= 50

        Complexity:
            Time: O(n * m) - We iterate through all possible pairs (i, j) 
                             which involves n * m checks. Each check involves
                             a multiplication and a modulo operation, which are O(1).
            Space: O(1) - We only use a few variables for counting and storing indices,
                          requiring constant extra space.
        """
        
        count = 0  # Initialize the counter for good pairs
        n = len(nums1) # Get the length of the first list
        m = len(nums2) # Get the length of the second list

        # Iterate through each index i for nums1 (from 0 to n-1)
        for i in range(n):
            # Get the current element from nums1
            num1_val = nums1[i] 
            
            # Iterate through each index j for nums2 (from 0 to m-1)
            for j in range(m):
                # Get the current element from nums2
                num2_val = nums2[j]
                
                # Calculate the required divisor based on nums2[j] and k
                # Since nums2[j] >= 1 and k >= 1, the divisor is always positive.
                divisor = num2_val * k
                
                # Check if the element from nums1 is divisible by the divisor.
                # This is the condition for a "good pair".
                # The modulo operator (%) returns 0 if num1_val is perfectly divisible by divisor.
                if num1_val % divisor == 0:
                    # If the condition is met, increment the count of good pairs
                    count += 1
        
        # Return the total number of good pairs found
        return count