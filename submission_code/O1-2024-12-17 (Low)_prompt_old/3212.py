class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        """
        We want to count the number of ways to split nums into contiguous subarrays so that
        no value appears in more than one subarray. In other words, each distinct number's
        occurrences must lie entirely within a single contiguous subarray.

        Key insight:
          • If a number x appears from index firstOcc[x] to lastOcc[x], we must NOT cut
            anywhere between those two indices, because that would separate duplicates of x
            into different subarrays.
          • Hence, for each distinct number x, the half-open interval [firstOcc[x], lastOcc[x])
            of "cut positions" is forbidden.
          • Overall, if we mark all forbidden cut positions and merge those intervals, the
            remaining cut positions are the ones we are free to choose. If there are 'M' such
            free positions, each can either be used or not used, giving 2^M ways to partition.
        
        Steps:
          1) Traverse nums to record firstOcc[x] and lastOcc[x] for each distinct x.
          2) Collect intervals [firstOcc[x], lastOcc[x]] and sort by their start.
          3) Merge overlapping intervals. Each interval [s, e] forbids cuts at s, s+1, ..., e-1.
             That is (e - s) forbidden positions.
          4) Let total_forbidden be the sum of merged-interval lengths. Then the number of
             allowed cut positions = (n-1) - total_forbidden.
          5) The answer is 2^(allowed positions) modulo 1e9+7.

        Examples:
          • [1,2,3,4] => all distinct => forbidden positions = 0 => allowed = 3 => 2^3 = 8
          • [1,1,1,1] => single number => interval [0,3] => forbids 0..2 => 3 forbidden => 0 allowed => 2^0 = 1
          • [1,2,1,3] => intervals: 1->[0,2], 2->[1,1], 3->[3,3]
            merged => [0,2], [3,3] => forbids positions 0..1 => 2 forbidden => allowed=3-2=1 => 2^1=2
        """

        MOD = 10**9 + 7
        n = len(nums)
        if n <= 1:
            return 1  # Only 1 way to partition a single-element array

        # Step 1: Find first and last occurrence of each distinct number
        from collections import defaultdict
        first_occ = {}
        last_occ = {}
        for i, val in enumerate(nums):
            if val not in first_occ:
                first_occ[val] = i
            last_occ[val] = i

        # Step 2: Build intervals and sort
        intervals = []
        for val in first_occ:
            start = first_occ[val]
            end = last_occ[val]
            intervals.append((start, end))
        intervals.sort()

        # Step 3: Merge overlapping intervals
        merged = []
        current_start, current_end = intervals[0]
        for i in range(1, len(intervals)):
            s, e = intervals[i]
            if s <= current_end:  # overlap
                current_end = max(current_end, e)
            else:
                # close off the previous interval
                merged.append((current_start, current_end))
                current_start, current_end = s, e
        merged.append((current_start, current_end))

        # Step 4: Count how many cut positions are forbidden
        # Each interval [s, e] forbids cuts s..(e-1) => length = e - s
        total_forbidden = 0
        for s, e in merged:
            total_forbidden += (e - s)

        allowed = (n - 1) - total_forbidden
        # Step 5: the number of good partitions = 2^allowed (mod 1e9+7)
        return pow(2, allowed, MOD)