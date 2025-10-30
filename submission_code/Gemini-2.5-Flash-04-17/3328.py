class Solution:
    def minOperations(self, k: int) -> int:
        # We are given an initial array [1] and a target sum k.
        # Operations allowed: increase an element by 1 (cost 1), duplicate an element (cost 1).
        # We want to find the minimum operations to make the sum of elements >= k.

        # The optimal strategy involves creating a homogeneous array [x, x, ..., x] with N elements.
        # This is because having elements of different values makes it less efficient to increase the sum.
        # For example, to add 'v' to the sum, we can increase an element by 'v' (cost 'v') or duplicate an element with value 'v' (cost 1).
        # Duplicating a larger value adds more to the sum per operation. To maximize this efficiency,
        # all elements should be the same (the largest possible value we decide to use).

        # To obtain N copies of value x from the initial [1]:
        # 1. Increase the value from 1 to x: x - 1 operations. Array becomes [x].
        # 2. Duplicate the element x N-1 times: N - 1 operations. Array becomes [x, ..., x].
        # Total operations: (x - 1) + (N - 1).

        # We need the sum x * N >= k.
        # For a chosen value x (the value of each element in the final array), the minimum
        # number of elements N that satisfies x * N >= k is ceil(k / x).
        # In integer arithmetic, ceil(a / b) for positive integers a, b is (a + b - 1) // b.
        # So, N = (k + x - 1) // x.

        # We want to minimize the total operations (x - 1) + (N - 1)
        # subject to N = (k + x - 1) // x, for x >= 1.
        # The function to minimize is C(x) = (x - 1) + ((k + x - 1) // x - 1).
        # This function is roughly x + k/x - 2, which has a minimum near x = sqrt(k).

        # Initialize the minimum operations found so far to infinity.
        min_ops = float('inf')

        # Iterate through possible values for x (the value of the elements in the final array)
        # starting from x = 1.
        # The cost function C(x) typically decreases for x < sqrt(k) and increases for x > sqrt(k).
        # We can iterate x from 1 upwards and stop when we are confident that
        # further increases in x will not result in a smaller cost.
        # The increasing term in the cost is (x - 1). The decreasing term is (ceil(k/x) - 1).
        # The total cost is C(x) = (x - 1) + (ceil(k/x) - 1).
        # Since ceil(k/x) >= 1 for x <= k (and we only need to check x up to k roughly),
        # the term (ceil(k/x) - 1) >= 0.
        # So, C(x) >= x - 1.
        # If the increasing term (x - 1) alone is already greater than or equal to the
        # current minimum operations (min_ops), then the total cost C(x) will be
        # >= x - 1 >= min_ops.
        # Thus, we can stop iterating when (x - 1) >= min_ops.

        x = 1
        while True:
            # Calculate the minimum number of elements N required for the current value x
            # N = ceil(k / x)
            N = (k + x - 1) // x

            # Calculate the total operations for this choice of x and N
            current_cost = (x - 1) + (N - 1)

            # Update the minimum operations found so far
            min_ops = min(min_ops, current_cost)

            # Check the stopping condition: if the increasing part (x-1)
            # is already greater than or equal to the best cost found so far,
            # the total cost cannot decrease further by increasing x.
            if (x - 1) >= min_ops:
                 break # The cost will not decrease further for larger x

            # Move to the next possible value for the element
            x += 1

        return min_ops