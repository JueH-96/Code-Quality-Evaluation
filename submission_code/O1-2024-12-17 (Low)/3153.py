class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        """
        We are given an array nums and a positive integer k. We can perform an
        operation any number of times, choosing indices i != j and simultaneously
        updating:
           nums[i] = nums[i] & nums[j]
           nums[j] = nums[i] | nums[j]
        Our goal is to pick k elements from the final array so as to maximize
        the sum of their squares, modulo 1e9+7.

        --------------------------------------------------------------------
        KEY OBSERVATIONS / SOLUTION OUTLINE:

        1) Bitwise Analysis and "Conservation" of Bits:
           Each operation (AND/OR) merely redistributes bits among the two chosen
           entries; it never creates or destroys bits.  In other words, if a certain
           bit is set in 'c' positions of the array initially, then in the final array,
           that same bit can appear overall at most 'c' times (across all positions).

        2) Only the Top k Matter:
           We only take the sum of squares of k chosen elements, so bits that end
           up outside this chosen subset don't help.  Thus, effectively, we can think
           of constructing (or "targeting") k final elements to be as large as
           possible—while respecting that each individual bit can appear at most c(b)
           times total, where c(b) is the count of how many numbers in nums had that
           bit set initially.

        3) Binary Presence (No "Multiple Copies" of the Same Bit in One Element):
           A single bit is either present (1) or absent (0) in a given final element.
           One element cannot hold multiple copies of the same bit.

        4) Distributing Bits to Maximize Sum of Squares:
           Since each bit can appear up to c(b) times across the final array and
           we only care about k final "slots," we would place that bit in
           min(c(b), k) of those slots.  The question is: which slots should we
           place them into to maximize the overall sum of squares?

           - Mathematical check: The incremental gain in the sum of squares from
             adding a higher bit to a bigger base value is larger than adding it
             to a smaller base.  Indeed, if you have an element e, adding 2^b
             changes the square from e^2 to (e + 2^b)^2 = e^2 + 2^(b+1)*e + 2^(2b).

             Because 2^(b+1)*e is larger if e is already large, it is more beneficial
             to put the bit into the largest slot(s) first.

        5) Greedy Algorithm (Descending Bits, Always Add to Largest Slots):
           a) We count how many times each bit b appears in the input (c[b]).
           b) We maintain k "final elements" e[0..k-1], all starting at 0.
           c) Iterate over bits from highest to lowest:
                - Let r = min(c[b], k).
                - We add bit b to the r largest elements in e so as to maximize sum of squares.
                - After each assignment, that element grows, so we re-sort to keep track
                  of which are now largest for the next bit.
           d) After processing all bits, we compute sum(e[i]^2) for i in 0..k-1.

        6) Complexity:
           - k can be up to 1e5, and we have up to ~30 relevant bit positions (since 1 <= nums[i] <= 1e9).
           - A straightforward implementation that re-sorts after each bit step is
             O(30 * k log k) ~ 30 * 1e5 * log(1e5).  In optimized Python or C++,
             this can be made to pass if carefully implemented.

        Let's implement this.
        """

        import sys
        input_data_mod = 10**9 + 7

        # 1) Count the occurrences c[b] of each bit b in nums
        #    We'll only need to check bits up to ~30 because nums[i] <= 1e9.
        bit_count = [0]*31
        for val in nums:
            for b in range(31):
                if (val >> b) & 1:
                    bit_count[b] += 1

        # 2) We only need to keep track of k "slots" for the final array
        #    since we will only take the sum of squares of k elements.
        #    Initialize all k slots to 0.
        #    If k == 0 (edge case not in the problem constraints), we'd return 0 directly.
        #    But constraints say 1 <= k <= len(nums), so we skip that edge check.
        e = [0]*k

        # 3) We'll process bits in descending order, always adding that bit
        #    to the largest slots (up to c[b] times).
        #    Then re-sort descending for the next bit.
        e.sort(reverse=True)
        for b in reversed(range(31)):
            # how many times can we place this bit?
            r = min(bit_count[b], k)
            # place bit b into the largest r elements in e
            for i in range(r):
                e[i] += (1 << b)
            # re-sort for the next iteration
            e.sort(reverse=True)

        # 4) Compute the sum of squares mod 1e9+7
        ans = 0
        for val in e:
            ans = (ans + (val % input_data_mod) * (val % input_data_mod)) % input_data_mod

        return ans