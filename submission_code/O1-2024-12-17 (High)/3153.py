class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        """
        We use a "bit-distribution" insight.  For each bit-position b (0..30),
        let x_b = how many numbers in nums have that bit set.  In the final
        array (after any number of the allowed AND/OR operations), we can set
        that bit in up to x_b different elements (because we cannot create new
        occurrences of a bit, only move or replicate it if two elements already
        have it).

        Ultimately, we only choose k elements for the sum-of-squares, so there
        is no benefit in creating more than k "large" elements.  We will
        maintain exactly k "aggregator" values that collect all the bits we
        want to use.  Each aggregator is an integer (initially 0), and setting
        a bit b in that aggregator increases its value by 2^b.

        To maximize the sum of squares of these k final values a_1,...,a_k,
        we note that adding a bit b to an aggregator a_i has incremental gain
        (a_i + 2^b)^2 - a_i^2 = 2^(2b) + (2^(b+1))*a_i.  This grows with a_i,
        so greedily we want to assign each available copy of bit b to whichever
        aggregator is currently largest.

        Algorithm:
          1) Count x_b for each b in [0..30] (how many nums have that bit).
          2) Keep k "aggregator" values in a max-heap (in Python, we store them
             as negative for convenience, so popping gives the largest actual value).
          3) Process bits from highest (30) down to lowest (0):
             - let c = x_b
             - if c >= k, then we can set bit b in all k aggregators (adding 2^b
               to each).  Since adding the same offset to every aggregator does
               not break the heap order, we just subtract (1<<b) from each
               stored negative value in O(k) time (no re-heapify needed).
             - else (c < k), we pop the largest aggregator c times, add 2^b,
               and push it back each time (this ensures each copy of bit b goes
               to the currently largest aggregator).
          4) At the end, we compute the sum of squares of these k values modulo
             1e9+7.
        """

        import sys
        import heapq
        
        MOD = 10**9 + 7
        
        # Count how many times each bit is set across all nums
        bit_count = [0]*31
        for num in nums:
            # Count bits in num
            # (A quick way is to loop over 31 bit positions,
            #  checking if that bit is set.)
            for b in range(31):
                if num & (1 << b):
                    bit_count[b] += 1
        
        # We maintain exactly k aggregator values (as a max-heap of size k).
        # If k == len(nums), that just means we combine them all, but in
        # general we only need k final "big" values.
        # Initialize them all to 0 (actual value), stored as 0 in negative form.
        aggregators = [0]*k
        heapq.heapify(aggregators)  # min-heap, but we will store negative values below.
        
        if k == 0:
            # Per constraints, k >= 1, so this should not happen.
            return 0
        
        # Convert to a "max-heap" by storing negative; pop will give largest actual value
        aggregators = [ -val for val in aggregators ]
        heapq.heapify(aggregators)
        
        # Process bits from high to low
        for b in reversed(range(31)):
            c = bit_count[b]
            add_val = (1 << b)
            
            if c >= k:
                # Add this bit to all k aggregators
                # subtract add_val from each negative aggregator since actual_value += add_val
                # The relative heap order does not change if we add the same offset to all.
                for i in range(k):
                    aggregators[i] -= add_val
                # No re-heapify needed because adding the same offset preserves the heap property
            else:
                # We can only place this bit in c aggregators, always pick the largest aggregator
                # each time for maximum gain.
                for _ in range(c):
                    largest_neg = heapq.heappop(aggregators)  # this is the most negative => largest actual
                    new_val_neg = largest_neg - add_val
                    heapq.heappush(aggregators, new_val_neg)
        
        # Now convert aggregators back to positive actual values
        final_vals = [ -v for v in aggregators ]
        
        # Sum of squares of these k aggregator values, mod 1e9+7
        ans = 0
        for val in final_vals:
            # (val % MOD) * (val % MOD) % MOD
            sq = (val % MOD) * (val % MOD) % MOD
            ans = (ans + sq) % MOD
        
        return ans