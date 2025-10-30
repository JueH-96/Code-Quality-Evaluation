from typing import List

class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        """
        Checks if it's possible to select two or more elements from nums
        such that their bitwise OR has at least one trailing zero.

        This condition is equivalent to the bitwise OR being an even number.
        The bitwise OR of a set of numbers is even if and only if all numbers
        in the set are even.
        Therefore, the problem reduces to checking if there are at least
        two even numbers in the input array.
        """
        
        even_count = 0
        for num in nums:
            # A number is even if it is divisible by 2.
            if num % 2 == 0:
                even_count += 1
                # If we have found two even numbers, we can form a valid pair.
                # Their bitwise OR will be even, thus having a trailing zero.
                if even_count >= 2:
                    return True
        
        # If the loop finishes, it means we found 0 or 1 even numbers,
        # which is not enough to select two or more.
        return False