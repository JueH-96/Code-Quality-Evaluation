import collections
from typing import List

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        """
        Determines the number of distinct three-digit even numbers that can be formed
        using the digits from the input array.
        """
        
        count = 0
        input_counts = collections.Counter(digits)

        # Iterate through all 3-digit even numbers from 100 to 998.
        # These numbers inherently do not have leading zeros.
        for num in range(100, 1000, 2):
            
            # Get the frequency of digits required to form the number `num`.
            required_counts = collections.Counter(map(int, str(num)))
            
            # Check if the multiset of required digits is a subset of the 
            # multiset of available input digits. The 'all' function with a 
            # generator expression is a concise and efficient way to perform this check.
            if all(input_counts[d] >= required_counts[d] for d in required_counts):
                count += 1
                
        return count