import collections
from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        """
        Calculates the bitwise XOR of all numbers that appear exactly twice in the input list.

        Args:
            nums: A list of integers where each number appears either once or twice.

        Returns:
            The bitwise XOR of all numbers appearing twice, or 0 if no number appears twice.
        """
        
        # Approach 1: Using a frequency counter (like collections.Counter or a dictionary)
        # Count occurrences of each number.
        # Iterate through the counts and XOR numbers with a count of 2.
        
        # counts = collections.Counter(nums)
        # xor_result = 0
        # for num, count in counts.items():
        #     if count == 2:
        #         xor_result ^= num
        # return xor_result

        # Approach 2: Using a set for single pass identification
        # Keep track of numbers seen once. When a number is encountered again,
        # it means it's a duplicate (appears twice), so XOR it into the result.
        
        seen_once = set()
        xor_result = 0
        
        for num in nums:
            # If we try to add a number already present in 'seen_once',
            # it means this is the second occurrence.
            if num in seen_once:
                 # This number appears twice, XOR it with the result.
                 # We don't need to remove it from seen_once because the problem
                 # guarantees numbers appear at most twice. If they could appear
                 # more times, we might need different logic.
                xor_result ^= num
            else:
                # First time seeing this number, add it to the set.
                seen_once.add(num)
                
        return xor_result