from math import ceil
from typing import List

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        # For a given number x, the number of "divisions" needed (each division is floor(x/4))
        # to reduce x to 0 is:
        #       f(x) = k   if 4^(k-1) <= x < 4^k.
        # For example, if x = 1,2,3 then 4^(0)=1 <= x < 4, so f(x)=1.
        # If x is between 4 and 15 then f(x)=2, and so on.
        #
        # In each operation we can process 2 numbers concurrently (each one undergoing one division),
        # so overall "steps" required equals sum_{x in array} f(x).
        # And the minimum number of operations is:
        #       ops = ceil(total_steps / 2)
        #
        # Given queries [l, r] that define the array with all integers from l through r,
        # we need to compute the sum of ops over all queries.
        
        # Precompute intervals for each possible k:
        # For each k >= 1, the set of integers with f(x)=k is [4^(k-1), 4^k - 1].
        # Note: Since r can be as high as 10^9, we need to compute k values until 4^(k-1) > 10^9.
        intervals = []
        p = 1  # 4^(k-1) for k=1 initially
        k = 1
        max_val = 10**9
        while p <= max_val:
            low = p
            high = 4 * p - 1  # because numbers in range [4^(k-1), 4^k-1] have f(x)=k
            intervals.append((k, low, high))
            k += 1
            p *= 4

        total_operations = 0
        # Process each query
        for q in queries:
            l, r = q
            total_steps = 0
            # For each interval corresponding to a fixed k (number of divisions needed) 
            # count how many numbers in the query [l, r] belong to that interval.
            for k, low, high in intervals:
                # If the interval for this f-value doesn't intersect with [l, r], skip.
                if r < low or l > high:
                    continue
                # Compute the intersection range:
                L = max(l, low)
                R = min(r, high)
                count = R - L + 1
                total_steps += k * count

            # Each operation can process 2 division steps.
            ops = (total_steps + 1) // 2
            total_operations += ops

        return total_operations