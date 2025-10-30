from typing import List
from collections import Counter

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Count the frequency of each number in nums
        freq = Counter(nums)
        # Initialize the maximum length to 0
        max_length = 0
        
        # Iterate over the unique numbers in nums
        for num in set(nums):
            # If the number is 1, it can always be part of the subset
            if num == 1:
                max_length = max(max_length, freq[num])
            else:
                # Start with the current number and try to build the pattern
                current_length = 0
                x = num
                while x in freq and freq[x] > 0:
                    # Increment the length of the current pattern
                    current_length += freq[x]
                    # Move to the next number in the pattern (x^2)
                    x *= x
                # Update the maximum length if the current pattern is longer
                max_length = max(max_length, current_length)
        
        return max_length

# Example usage:
# sol = Solution()
# print(sol.maximumLength([5,4,1,2,2]))  # Output: 3
# print(sol.maximumLength([1,3,2,4]))    # Output: 1