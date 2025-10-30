import math

class Solution:
    def minOperations(self, k: int) -> int:
        # If k is 1, the sum of the initial array [1] is already >= k,
        # so no operations are needed.
        if k == 1:
            return 0

        # We aim to reach a state where the array has 'n' elements,
        # and all elements have the same value 'x'.
        # The sum will then be n * x. We need this sum to be >= k.
        #
        # The operations to achieve this state are:
        # 1. Increase the initial element [1] to [x]. This costs (x - 1) operations.
        # 2. Duplicate the element 'x' (n - 1) times to get 'n' copies. This costs (n - 1) operations.
        # Total operations = (x - 1) + (n - 1) = x + n - 2.
        #
        # To minimize operations for a chosen 'x', we need to find the smallest 'n'
        # such that n * x >= k. This 'n' is ceil(k / x).
        # In integer arithmetic, ceil(a / b) can be computed as (a + b - 1) // b.
        # So, n = (k + x - 1) // x.
        #
        # We will iterate through possible values of 'x' and calculate the corresponding 'n'
        # and the total operations (x + n - 2), keeping track of the minimum.
        #
        # The function f(x) = x + k/x (approximately) tends to be minimized when x is near sqrt(k).
        # Since k is up to 10^5, sqrt(k) is about 316.
        # Iterating 'x' from 1 up to int(sqrt(k)) + a small buffer (e.g., 2) is sufficient
        # to find the optimal 'x'. This ensures an O(sqrt(k)) time complexity.

        min_total_ops = float('inf')

        # The loop for 'x' can go from 1 up to sqrt(k) + a small offset.
        # Adding '2' to math.sqrt(k) gives a safe upper bound for checking 'x'.
        # For example, if k=4, sqrt(k)=2. max_x_to_check becomes 2+2=4.
        # The loop will check x=1,2,3,4.
        max_x_to_check = int(math.sqrt(k)) + 2

        for x in range(1, max_x_to_check + 1):
            # Calculate the minimum number of elements 'n' required for the current 'x'
            # to satisfy the sum condition (n * x >= k).
            n = (k + x - 1) // x
            
            # Calculate the total operations for this specific (x, n) pair.
            current_ops = (x - 1) + (n - 1)
            
            # Update the minimum operations found so far.
            min_total_ops = min(min_total_ops, current_ops)
            
        return min_total_ops