from typing import List

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        # Precompute powers of 4
        # ops(x) = k where 4^(k-1) <= x < 4^k for x >= 1.
        # The number of individual reduction steps for x is k.
        # For x <= 10^9, the maximum ops value is 15 (since 4^14 <= 10^9 < 4^15).
        # We need powers of 4 up to 4^15 to define the ranges for ops values 1 to 15.
        # pows4[k] will store 4^k. We need pows4[0] up to pows4[15].
        pows4 = [1]
        for _ in range(15):
            pows4.append(pows4[-1] * 4)
        # pows4 now contains [4^0, 4^1, ..., 4^15]

        def calc_sum_ops(N):
            """
            Calculates the sum of ops(x) for x from 1 to N.
            sum_{x=1}^{N} ops(x)
            """
            if N <= 0:
                return 0

            total_sum = 0
            # ops(x) = k for x in the range [pows4[k-1], pows4[k]-1] for k >= 1.
            # We want to compute sum_{x=1}^{N} ops(x) = sum_{k=1}^{max_ops} k * (count of x in [1, N] with ops(x)=k)
            # The count of x in [1, N] with ops(x)=k is the number of integers in the intersection of [1, N] and [pows4[k-1], pows4[k]-1].
            # This intersection is [max(1, pows4[k-1]), min(N, pows4[k]-1)].
            # Since pows4[k-1] is 4^(k-1) which is >= 1 for k >= 1, the range is [pows4[k-1], min(N, pows4[k]-1)].

            # Iterate through possible ops values k, from 1 upwards.
            # The maximum relevant ops value for N <= 10^9 is 15.
            # We iterate k from 1 up to 15.
            for k in range(1, 16): # k represents the ops value
                low = pows4[k-1]
                high = pows4[k] - 1 # pows4[k] is 4^k. e.g., for k=15, pows4[15] = 4^15.

                # The segment of numbers [low, high] has ops value k.
                # We need to count how many numbers in [1, N] fall into this segment.
                # This is the count of integers in the range [low, min(N, high)].
                start = low
                end = min(N, high)

                # If the effective range [start, end] is empty or invalid (start > end)
                if start > end:
                    # If N is less than the start of the current ops group [low, high],
                    # it implies N is less than 4^(k-1). All subsequent groups [4^j, 4^(j+1)-1] for j >= k
                    # will have start values >= 4^k > 4^(k-1) > N. So no more intersections are possible.
                    if N < start:
                         break
                    # If start <= end, continue calculation.

                # The number of integers in the range [start, end] is end - start + 1.
                count = end - start + 1
                total_sum += k * count

            return total_sum

        total_operations_sum = 0
        for l, r in queries:
            # The initial array for a query [l, r] contains numbers from l to r inclusive.
            # The sum of ops values for this array is sum_{x=l}^{r} ops(x).
            # This sum can be calculated using the prefix sum approach:
            # sum_{x=l}^{r} ops(x) = (sum_{x=1}^{r} ops(x)) - (sum_{x=1}^{l-1} ops(x))
            sum_ops_l_r = calc_sum_ops(r) - calc_sum_ops(l - 1)

            # The minimum number of operations to reduce all elements to zero is
            # ceil(sum(ops(x)) / 2).
            # Integer division for ceiling: (S + 1) // 2.
            total_operations_sum += (sum_ops_l_r + 1) // 2

        return total_operations_sum