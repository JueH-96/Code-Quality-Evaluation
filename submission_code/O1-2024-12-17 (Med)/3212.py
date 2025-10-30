class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        """
        We say a partition of nums into contiguous subarrays is 'good' if
        no integer appears in more than one subarray.

        Equivalently, if a value x appears in positions [Lx..Rx] of nums,
        then all those positions must lie entirely within exactly one of
        the partition's subarrays—so we cannot 'cut' across x's interval.

        Therefore, any place i (0 <= i < n-1) between i and i+1 can be a valid
        "cut" if there is no value's interval [Lx..Rx] that straddles i (i.e.
        Lx <= i and Rx >= i+1).  If a boundary is not crossed by any interval,
        it is a valid place to cut (or not cut).

        The total count of good partitions is then 2^(count_of_valid_cuts),
        because each valid cut can independently either be used or not used.

        Steps to implement:

        1) Traverse nums once while tracking for each distinct value x its
           first occurrence (Lx) and last occurrence (Rx).
        2) Build a list "starts" such that starts[Lx] holds Rx.  (For each Lx,
           we append the corresponding Rx to starts[Lx].)
        3) Define coverage[i] as the maximum 'end' of any interval that begins
           at or before i.  We maintain a running max while iterating i from 0
           to n-1, updating it with any intervals that start at i.
        4) A boundary between i and i+1 (for i in [0..n-2]) is valid if
           coverage[i] < i+1.  Count how many such valid boundaries exist.
        5) The answer is 2^(that count), modulo 10^9+7.

        Examples:
          - [1,2,3,4]: each value has an interval [i..i], so no boundary is crossed.
            All 3 boundaries are valid => 2^3 = 8.
          - [1,1,1,1]: the single value 1 has interval [0..3], so it crosses all
            boundaries in [0..2].  Hence none are valid => 2^0 = 1.
          - [1,2,1,3]: intervals are 1->[0..2], 2->[1..1], 3->[3..3].
            Only the boundary i=2 is not crossed by [0..2], so exactly 1 valid cut => 2^1=2.
        """

        import sys
        mod = 10**9 + 7

        n = len(nums)
        if n == 1:
            return 1  # Only one way to partition a single element

        # Step 1: find first and last occurrence for each distinct value
        first_occ = {}
        last_occ = {}
        for i, val in enumerate(nums):
            if val not in first_occ:
                first_occ[val] = i
            last_occ[val] = i

        # Step 2: build 'starts' to keep track of intervals [Lx..Rx]
        starts = [[] for _ in range(n)]
        for val in first_occ:
            Lx = first_occ[val]
            Rx = last_occ[val]
            starts[Lx].append(Rx)

        # Step 3: coverage[i] = max end of intervals that have started
        coverage = [0]*n
        max_end = -1
        for i in range(n):
            # update max_end with intervals that start at i
            for r in starts[i]:
                if r > max_end:
                    max_end = r
            coverage[i] = max_end

        # Step 4: count valid boundaries
        valid_cuts = 0
        for i in range(n-1):
            if coverage[i] < i+1:
                valid_cuts += 1

        # Step 5: answer = 2^(valid_cuts) mod 10^9+7
        return pow(2, valid_cuts, mod)