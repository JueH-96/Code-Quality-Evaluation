import math # Not strictly needed for the final implementation but good practice

class Solution:
    """
    Solves the problem of finding the minimum operations to reach a sum >= k.
    """
    def minOperations(self, k: int) -> int:
        """
        Calculates the minimum number of operations to make the sum of elements in an array greater than or equal to k.
        The initial array is nums = [1].
        Allowed operations:
        1. Choose any element and increase its value by 1 (cost: 1 operation).
        2. Duplicate any element and add it to the end (cost: 1 operation).

        The core idea is that the optimal final array structure consists of 'n' identical elements, each with value 'x'.
        This is because duplicating larger elements or increasing any element contributes positively or neutrally compared 
        to duplicating smaller elements when aiming for a target sum efficiently.
        
        To reach a state where the array is [x, x, ..., x] (n times) starting from [1]:
        - We need (x - 1) 'increase' operations to change the initial 1 to x.
        - We need (n - 1) 'duplicate' operations to get n copies of x from a single x.
        The total number of operations is ops = (x - 1) + (n - 1).

        We need the sum of the final array to be at least k. For the array [x, x, ..., x] (n times), the sum is n * x.
        So, we need to satisfy the condition n * x >= k.

        The problem reduces to finding positive integers x >= 1 and n >= 1 such that n * x >= k, 
        while minimizing the total operations: ops = (x - 1) + (n - 1).

        We can approach this by iterating through possible values for the final element 'x'. For a chosen 'x', 
        the minimum number of elements 'n' required to satisfy n * x >= k is n = ceil(k / x). 
        This is because n must be an integer.

        So, for a given x, the minimum operations needed is ops(x) = (x - 1) + (ceil(k / x) - 1).
        We need to find the minimum value of ops(x) over all possible x >= 1.

        Args:
            k: The target sum (a positive integer, 1 <= k <= 10^5).

        Returns:
            The minimum number of operations required.
        """
        
        # Base case: If k is 1, the initial array [1] already has sum = 1, which is >= k.
        # No operations are needed.
        if k == 1:
            return 0

        # We aim to minimize ops(x) = (x - 1) + (ceil(k / x) - 1).
        # Let's calculate the number of operations for x = 1 as an initial baseline minimum.
        # If x = 1, then n = ceil(k / 1) = k.
        # ops(1) = (1 - 1) + (k - 1) = 0 + k - 1 = k - 1.
        min_ops = k - 1

        # Now, we iterate through possible values of the element 'x', starting from x = 2.
        # The function ops(x) = x + ceil(k / x) - 2 generally decreases and then increases (it's U-shaped or convex-like).
        # Due to this property, we can stop iterating as soon as the value of ops(x) starts increasing 
        # or becomes equal to the minimum value found so far. Any further iterations would yield larger values.
        # The loop could theoretically go up to x = k (since for x=k, n=1, ops=k-1), 
        # but the break condition ensures it terminates much earlier, typically around x = sqrt(k).
        for x in range(2, k + 1): 
            # Calculate the minimum count 'n' required for the current value 'x'.
            # We use integer arithmetic for ceiling division: ceil(a / b) = (a + b - 1) // b
            # This formula works correctly for positive integers a and b.
            # Since k >= 1 and x >= 2 in this loop, k > 0 and x > 0 are satisfied.
            n = (k + x - 1) // x
            
            # Calculate the total number of operations for this pair (x, n).
            current_ops = (x - 1) + (n - 1)

            # Check the early stopping condition based on the U-shaped property of ops(x).
            # If the current number of operations is not less than the minimum found so far,
            # it implies we have reached or passed the minimum point of the function ops(x).
            if current_ops >= min_ops:
                # The minimum value has been found (or passed). Stop the search.
                break 
            
            # If current_ops is smaller than min_ops, we've found a new best value. Update min_ops.
            min_ops = current_ops

        # After the loop terminates (either by completing the range or breaking early), 
        # min_ops holds the minimum number of operations required.
        return min_ops