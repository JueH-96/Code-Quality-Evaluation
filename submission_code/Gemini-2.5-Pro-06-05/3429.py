import collections

class Solution:
    def isArraySpecial(self, nums: list[int]) -> bool:
        """
        Checks if an array is special.
        An array is special if every pair of its adjacent elements has different parity.
        """
        
        # An array with 0 or 1 elements is considered special as there are
        # no adjacent pairs to violate the condition.
        if len(nums) <= 1:
            return True
            
        # Iterate from the first element up to the second-to-last element.
        for i in range(len(nums) - 1):
            # Check if the current element and the next one have the same parity.
            # The parity of a number n is n % 2.
            # If (a % 2) == (b % 2), they have the same parity.
            if (nums[i] % 2) == (nums[i+1] % 2):
                # If we find such a pair, the array is not special.
                return False
        
        # If the loop completes without finding any pair with the same parity,
        # the array is special.
        return True