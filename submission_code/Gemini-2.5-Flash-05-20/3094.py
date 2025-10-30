import collections
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # Step 1: Count the frequency of each number in the array.
        # collections.Counter is an efficient way to do this.
        counts = collections.Counter(nums)
        
        total_operations = 0
        
        # Step 2: Iterate through the frequencies of each unique number.
        for num_val, count in counts.items():
            # Step 3: Handle the impossible case: If any number appears only once,
            # it's impossible to remove it, as operations require at least 2 or 3 elements.
            if count == 1:
                return -1
            
            # Step 4: Calculate the minimum operations for the current 'count'.
            # We want to represent 'count' as a sum of 2s and 3s, minimizing the number of terms.
            # This is equivalent to finding the minimum number of operations of size 2 or 3
            # to remove 'count' items.
            
            # The strategy is to use as many 3-element operations as possible because they remove
            # the most elements per operation.
            #
            # The number of operations needed for a count 'k' (where k >= 2) can be calculated
            # using integer ceiling division: ceil(k / 3).
            # In Python, for positive integers a and b, ceil(a/b) can be computed as (a + b - 1) // b.
            # Here, b=3, so (k + 3 - 1) // 3 which simplifies to (k + 2) // 3.
            #
            # Let's verify (count + 2) // 3 for different remainders:
            # - If count % 3 == 0 (e.g., count = 3, 6):
            #   (3 + 2) // 3 = 5 // 3 = 1  (correct, 1 op of 3)
            #   (6 + 2) // 3 = 8 // 3 = 2  (correct, 2 ops of 3)
            # - If count % 3 == 1 (e.g., count = 4, 7):
            #   (4 + 2) // 3 = 6 // 3 = 2  (correct, 2 ops of 2, i.e., 2+2)
            #   (7 + 2) // 3 = 9 // 3 = 3  (correct, 1 op of 3 and 2 ops of 2, i.e., 3+2+2)
            # - If count % 3 == 2 (e.g., count = 2, 5):
            #   (2 + 2) // 3 = 4 // 3 = 1  (correct, 1 op of 2)
            #   (5 + 2) // 3 = 7 // 3 = 2  (correct, 1 op of 3 and 1 op of 2, i.e., 3+2)
            # This formula correctly gives the minimum operations for any count k >= 2.
            
            total_operations += (count + 2) // 3
            
        # Step 5: Return the accumulated total operations.
        return total_operations