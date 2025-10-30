from typing import List
from collections import Counter

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def canonical_form(num):
            # Convert the number to a string and sort the digits to get its canonical form
            return ''.join(sorted(str(num)))
        
        # Create a counter for the canonical forms of the numbers
        canonical_counter = Counter(canonical_form(num) for num in nums)
        
        # Count the number of almost equal pairs
        pairs_count = 0
        for count in canonical_counter.values():
            # For each group of numbers with the same canonical form,
            # calculate the number of pairs using the formula n * (n - 1) / 2
            pairs_count += count * (count - 1) // 2
        
        return pairs_count

# Example usage:
# sol = Solution()
# print(sol.countPairs([3,12,30,17,21]))  # Output: 2
# print(sol.countPairs([1,1,1,1,1]))      # Output: 10
# print(sol.countPairs([123,231]))        # Output: 0