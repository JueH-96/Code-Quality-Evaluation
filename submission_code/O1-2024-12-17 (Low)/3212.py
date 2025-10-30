class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        """
        We say a partition of the array into one or more contiguous subarrays is "good" 
        if no two subarrays share a common value. Equivalently, each distinct number's 
        set of occurrences must lie fully in exactly one of the partition's subarrays.
        
        A key observation is that if a number x appears from index L to R (0-based),
        we cannot place any subarray boundary strictly between L and R. Hence each
        distinct number defines an interval [L, R] that must lie fully in one subarray.
        
        The partition is then governed by the union of these intervals. If intervals
        overlap, they merge into one larger "block," because no boundary is allowed 
        to cut through any of those intervals. After merging all intervals, we end 
        up with m disjoint blocks. Each block must be wholly contained in a single 
        subarray, but consecutive blocks may either remain separate or be merged 
        together (since there's no number spanning across disjoint blocks).

        Therefore, if there are m disjoint blocks in total, there are 2^(m-1) ways
        to combine/keep-split these blocks to form subarrays. (Between each pair of 
        adjacent blocks, we can choose to "put a cut" or "not put a cut".)

        Steps:
         1) Find for each distinct value x its minOcc[x] and maxOcc[x].
         2) Convert each into an interval [minOcc[x], maxOcc[x]].
         3) Sort these intervals by their start.
         4) Merge overlapping intervals.
         5) Let m be the number of merged intervals; the result is 2^(m-1) mod (1e9+7).

        Examples:
         - [1,2,3,4] -> each number is by itself, forming 4 disjoint intervals -> answer = 2^(4-1)=8
         - [1,1,1,1] -> number 1 spans [0, 3], only 1 interval -> answer = 2^(1-1)=1
         - [1,2,1,3] -> intervals: 1 -> [0,2], 2 -> [1,1], 3 -> [3,3].
                        Merged -> [0,2] and [3,3], so 2 disjoint blocks -> 2^(2-1)=2
        """
        import sys
        mod = 10**9 + 7

        # Step 1: find minOcc and maxOcc for each distinct number
        minOcc = {}
        maxOcc = {}
        n = len(nums)
        for i, val in enumerate(nums):
            if val not in minOcc:
                minOcc[val] = i
            maxOcc[val] = i  # always update

        # Step 2: build intervals
        intervals = []
        for val in minOcc:
            intervals.append((minOcc[val], maxOcc[val]))

        # Step 3: sort by start index
        intervals.sort(key=lambda x: x[0])

        # Step 4: merge overlapping intervals
        merged = []
        curr_start, curr_end = intervals[0]
        for i in range(1, len(intervals)):
            s, e = intervals[i]
            if s <= curr_end:  # overlap
                curr_end = max(curr_end, e)
            else:
                # push the current interval and start a new one
                merged.append((curr_start, curr_end))
                curr_start, curr_end = s, e
        # push the last interval
        merged.append((curr_start, curr_end))

        # m is the number of merged disjoint intervals
        m = len(merged)

        # Step 5: number of ways = 2^(m-1)
        # (Note: if m=1, answer = 1 = 2^0)
        if m == 1:
            return 1
        # fast exponentiation or built-in pow
        return pow(2, m-1, mod)