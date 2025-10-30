import collections
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        
        # The problem is about removing groups of 2 or 3 identical elements.
        # The order of elements or their values do not matter, only their frequencies.
        # So, the first step is to count the frequency of each number.
        counts = collections.Counter(nums)
        
        # Initialize the total number of operations.
        total_operations = 0
        
        # We can process the count of each unique number independently.
        for count in counts.values():
            
            # If a number appears only once, it's impossible to remove it
            # because both allowed operations require at least two identical elements.
            # In this case, we can never empty the array.
            if count == 1:
                return -1
            
            # For a count 'c' >= 2, we want to find the minimum number of
            # operations to clear all 'c' elements using groups of 2 or 3.
            # This is equivalent to finding non-negative integers 'a' and 'b'
            # such that c = 2*a + 3*b, where we want to minimize a + b.
            #
            # The minimum number of operations required is the ceiling of count / 3.
            # For example:
            # count = 2 -> ceil(2/3) = 1 op (a group of 2)
            # count = 3 -> ceil(3/3) = 1 op (a group of 3)
            # count = 4 -> ceil(4/3) = 2 ops (two groups of 2)
            # count = 5 -> ceil(5/3) = 2 ops (one group of 3, one group of 2)
            # count = 6 -> ceil(6/3) = 2 ops (two groups of 3)
            #
            # The ceiling function ceil(x/y) for positive integers can be
            # calculated using integer division as (x + y - 1) // y.
            # So, ceil(count / 3) is (count + 3 - 1) // 3 = (count + 2) // 3.
            total_operations += (count + 2) // 3
            
        return total_operations