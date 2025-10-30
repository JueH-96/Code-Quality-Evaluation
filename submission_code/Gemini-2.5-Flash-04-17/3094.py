from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        """
        Calculates the minimum number of operations required to make the array empty.

        Operations:
        - Delete 2 equal elements.
        - Delete 3 equal elements.

        Args:
            nums: A list of positive integers.

        Returns:
            The minimum number of operations, or -1 if it's impossible.
        """
        # Count the frequency of each number
        # Since operations only work on elements of equal value,
        # the problem can be solved independently for each distinct number.
        counts = Counter(nums)

        total_operations = 0

        # Iterate through the counts of each distinct number
        for count in counts.values():
            # If any number appears exactly once, it's impossible to remove it
            # using operations of size 2 or 3.
            if count == 1:
                return -1

            # For any count > 1, the minimum operations required to reduce this
            # count to zero using operations of size 2 and 3 can be found.
            # Let the count be c. We want to find non-negative integers n2 and n3
            # such that 2*n2 + 3*n3 = c, minimizing n2 + n3.

            # The optimal strategy is to use as many operations of size 3 as possible.
            # Let k = count // 3 be the number of full groups of 3 we can form initially.
            # Let remainder = count % 3.
            #
            # - If remainder == 0 (count = 3k):
            #   We can use k operations of size 3. Total ops = k = count // 3.
            #
            # - If remainder == 1 (count = 3k + 1):
            #   Since count > 1, k must be at least 1 (e.g., count=4 when k=1).
            #   Using k ops of 3 leaves 1, which cannot be removed by 2s or 3s.
            #   We must use fewer 3s. If we use k-1 ops of 3, the remaining count is
            #   (3k + 1) - 3*(k-1) = 3k + 1 - 3k + 3 = 4.
            #   This remaining 4 can be removed by two operations of size 2 (2+2=4).
            #   Total operations = (k - 1) + 2 = k + 1.
            #   Since k = (count - 1) // 3, operations = (count - 1) // 3 + 1.
            #   Alternatively, this is count // 3 + 1 when count % 3 == 1.
            #
            # - If remainder == 2 (count = 3k + 2):
            #   Since count >= 2, k can be 0 (e.g., count=2 when k=0).
            #   Using k ops of 3 leaves a remainder of 2.
            #   This remaining 2 can be removed by one operation of size 2.
            #   Total operations = k + 1.
            #   Since k = (count - 2) // 3, operations = (count - 2) // 3 + 1.
            #   Alternatively, this is count // 3 + 1 when count % 3 == 2.

            # A compact way to represent the minimum operations for any count > 1:
            # If count % 3 == 0, ops = count // 3
            # If count % 3 == 1, ops = count // 3 + 1
            # If count % 3 == 2, ops = count // 3 + 1
            # This pattern can be summarized using integer division as (count + 2) // 3.
            # Let's verify:
            # For count=3k: (3k + 2) // 3 = k (since 3k <= 3k+2 < 3k+3)
            # For count=3k+1: (3k+1 + 2) // 3 = (3k+3) // 3 = k+1
            # For count=3k+2: (3k+2 + 2) // 3 = (3k+4) // 3 = k+1 (since 3k+3 <= 3k+4 < 3k+6)
            # This formula works for all count >= 2.

            operations_for_this_count = (count + 2) // 3
            total_operations += operations_for_this_count

        # Return the total minimum operations across all distinct numbers
        return total_operations