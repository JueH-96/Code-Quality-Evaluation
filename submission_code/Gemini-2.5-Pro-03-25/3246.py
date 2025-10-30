from typing import List

class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        """
        Checks if it is possible to select two or more elements in the array
        such that the bitwise OR of the selected elements has at least one
        trailing zero in its binary representation.

        Args:
            nums: A list of positive integers.

        Returns:
            True if such a selection is possible, False otherwise.

        Explanation:
        1. Trailing Zeros and Even Numbers: A number has at least one trailing zero 
           in its binary representation if and only if the number is even. This is 
           because the least significant bit (LSB) of an even number is 0, and the 
           LSB of an odd number is 1.

        2. Bitwise OR and Evenness: We need to determine if it's possible to select 
           a subset of `nums` with size >= 2 such that their bitwise OR is even.
           Let the selected subset be S = {s1, s2, ..., sk}, where k >= 2.
           We want the result `R = s1 | s2 | ... | sk` to be even.
           R is even if and only if its LSB is 0.

        3. LSB of Bitwise OR: The LSB of a bitwise OR operation is the bitwise OR 
           of the LSBs of the operands.
           LSB(R) = LSB(s1 | s2 | ... | sk) = LSB(s1) | LSB(s2) | ... | LSB(sk).

        4. Condition for Even OR: For R to be even, we need LSB(R) = 0. This means:
           LSB(s1) | LSB(s2) | ... | LSB(sk) = 0.
           The bitwise OR of bits is 0 if and only if all the individual bits are 0.
           Therefore, we must have LSB(si) = 0 for all i from 1 to k.

        5. LSB and Even Numbers: LSB(si) = 0 means that the number si must be even.

        6. Conclusion: The problem requires finding if we can select a subset S of 
           `nums` such that |S| >= 2 and all elements in S are even. This is only 
           possible if the original array `nums` contains at least two even numbers.
           If we find two even numbers, say `e1` and `e2`, we can select the subset 
           {e1, e2}. Their bitwise OR, `e1 | e2`, will have LSB(e1 | e2) = LSB(e1) | LSB(e2) = 0 | 0 = 0,
           meaning `e1 | e2` is even and has at least one trailing zero.

        7. Algorithm: Count the number of even elements in `nums`. If the count is 
           2 or greater, return True. Otherwise, return False. We can optimize by
           returning True as soon as we find the second even number.
        """
        
        even_count = 0  # Initialize counter for even numbers found so far
        
        # Iterate through each number in the input list
        for num in nums:
            # Check if the current number is even. 
            # An integer is even if its remainder when divided by 2 is 0.
            if num % 2 == 0:
                even_count += 1  # Increment the count if the number is even
                
                # Optimization: If we have found two even numbers, we know it's possible
                # to select two even numbers whose OR will be even.
                # We can stop iterating and return True immediately.
                if even_count >= 2: # Check if we have found at least two even numbers
                    return True

        # If the loop completes without finding two or more even numbers
        # (i.e., even_count is 0 or 1), it's impossible to select two or more
        # elements whose bitwise OR is even.
        return False