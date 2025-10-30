import collections
from typing import List
import math # math.ceil can be used as an alternative, but integer division is typically preferred.

class Solution:
    """
    Solves the problem of finding the minimum operations to empty an array
    by removing pairs or triplets of equal elements.
    """
    def minOperations(self, nums: List[int]) -> int:
        """
        Calculates the minimum number of operations required to make the array empty.

        The allowed operations on the array are:
        1. Choose two elements with equal values and delete them.
        2. Choose three elements with equal values and delete them.

        The goal is to make the array empty using the minimum total number of operations.

        Args:
            nums: A list of positive integers. The input array.

        Returns:
            The minimum number of operations required to empty the array,
            or -1 if it is impossible to empty the array using the given operations.
        """
        # Step 1: Count the frequency of each number in the input array.
        # Using collections.Counter is an efficient way to get the counts of all
        # elements in O(N) time, where N is the length of the array.
        # Example: nums = [2, 3, 3, 2, 2, 4, 2, 3, 4] -> counts = {2: 4, 3: 3, 4: 2}
        counts = collections.Counter(nums)

        total_operations = 0

        # Step 2: Iterate through the frequency count of each unique number present in the array.
        # The operations for different numbers are independent. We can calculate the
        # minimum operations required for each number based on its count and sum them up.
        for count in counts.values():
            # Step 3: Check for impossibility.
            # If a number appears exactly once (count == 1), it cannot be removed
            # because both allowed operations require at least two equal elements.
            # In this situation, it's impossible to empty the array.
            if count == 1:
                return -1

            # Step 4: Calculate the minimum operations for the current count `c`.
            # We need to express the count `c` (where c >= 2) as a sum of 2s and 3s:
            # c = 2 * op2 + 3 * op3, where op2 and op3 are the number of times we
            # perform the operation of removing 2 and 3 elements, respectively.
            # We want to minimize the total number of operations: op = op2 + op3.
            #
            # It can be shown that for any count c >= 2, it's always possible to
            # clear the elements, and the minimum number of operations required is
            # mathematically equivalent to ceil(c / 3).
            #
            # We can calculate ceil(c / 3) efficiently using integer division:
            # ceil(c / 3) = (c + 2) // 3 for positive integers c.
            #
            # Let's verify this formula for small counts >= 2:
            # count = 2: ceil(2/3) = 1. Formula: (2 + 2) // 3 = 4 // 3 = 1. (Need one op of size 2)
            # count = 3: ceil(3/3) = 1. Formula: (3 + 2) // 3 = 5 // 3 = 1. (Need one op of size 3)
            # count = 4: ceil(4/3) = 2. Formula: (4 + 2) // 3 = 6 // 3 = 2. (Need two ops of size 2)
            # count = 5: ceil(5/3) = 2. Formula: (5 + 2) // 3 = 7 // 3 = 2. (Need one op size 3, one op size 2)
            # count = 6: ceil(6/3) = 2. Formula: (6 + 2) // 3 = 8 // 3 = 2. (Need two ops of size 3)
            #
            # The formula holds.
            operations_for_this_number = (count + 2) // 3
            
            # Alternatively using math.ceil:
            # operations_for_this_number = math.ceil(count / 3) 

            # Step 5: Add the minimum operations needed for this number's count
            # to the overall total number of operations.
            total_operations += operations_for_this_number

        # Step 6: Return the total minimum operations.
        # If the loop completed without returning -1, it implies all numbers
        # had counts of 2 or more, making it possible to empty the array.
        # The value `total_operations` now holds the minimum total operations required.
        return total_operations