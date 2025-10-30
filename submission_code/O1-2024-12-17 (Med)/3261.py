class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        """
        We want to partition the array into at most (k) merges (so at least (n-k) segments)
        in such a way that the bitwise-OR of the segment-ANDs is minimized.

        ----------------------------------------------------------------------------------------------------------------
        KEY INSIGHT / RESTATEMENT:

        - Each "merge" operation replaces two consecutive elements x, y by (x & y).  
        - After up to k merges, we end up with an array of length s in the range [n-k, n].  
          In other words, the final partition has between n-k and n "segments."  

        - Equivalently, think of splitting the original array into s contiguous segments
          (for s in [n-k, n]), and each segment is replaced by the bitwise AND of its elements.
          Then we take the bitwise OR of all those segment-ANDs.  Our goal is to minimize that OR.

        - We will binary-search on the possible target "M" for that OR.  For a candidate M,
          we check feasibility: "Can we form a partition of the array into at least (n-k) segments
          where each segment's AND ≤ M?"

          If it is feasible to form ≥ (n-k) segments with ANDs ≤ M, then after merging (n - number_of_segments)
          times, we use ≤ k merges, meaning M is a valid upper bound for the final OR. Otherwise, it's not feasible.

        ----------------------------------------------------------------------------------------------------------------
        CHECKING FEASIBILITY (CAN WE GET ≥ (n-k) SEGMENTS WITH SEGMENT-AND ≤ M?):

        Observe that "segment-AND ≤ M" means that, in bitwise terms, no segment-AND has a bit that is not in M.
        Another way: segment-AND & (~M) = 0.

        The tricky part is efficiently finding the maximum number of valid segments (where each segment AND ≤ M).
        A naïve O(n^2) approach is too large for n=1e5.

        We can do a "sparse table" for O(1) subarray-AND queries, then greedily find the smallest valid subarray
        starting from each index.  This is done via a binary search for each starting position.  In total, that
        yields O(n log n) for feasibility checks.  Repeated inside a ~30-step binary search on M gives
        O(n log n * log(max_num)) which can be made to pass in fast languages (C++).  In Python, it is borderline
        but often still acceptable with efficient implementation.

        Steps for feasibility(M):
          1) We'll scan from left to right (index i).
          2) From i, binary-search for the minimal j >= i such that AND of [i..j] ≤ M.
             If none, feasibility = False immediately.
             Otherwise, that subarray becomes one segment.  Then set i = j+1, repeat.
          3) Count how many such segments we form.  If the count of segments ≥ (n-k), feasibility = True,
             else False.

        Then do a global binary search on M from 0..(bitwise OR of all nums), and return the minimum feasible M.

        ----------------------------------------------------------------------------------------------------------------
        IMPLEMENTATION DETAILS:

        1) Precompute hi = OR of all elements in nums.  That is the maximum possible OR (upper bound).
        2) Build a sparse table st[][] for fast subarray-AND queries:
             - st[i][0] = nums[i].
             - st[i][p] = st[i][p-1] & st[i + 2^(p-1)][p-1].
           Then AND of [L..R] is st[L][p] & st[R - 2^p + 1][p], where p = floor(log2(R-L+1)).

        3) Define get_and(L, R) that returns the AND of nums[L..R] using st in O(1).

        4) Define feasible(M):
             - Initialize count_segments = 0
             - i = 0
             - While i < n:
                 * If get_and(i, n-1) > M, then even the largest subarray [i..end] is invalid => fail (return False).
                 * Binary-search on j in [i..n-1] to find the minimal j s.t. get_and(i,j) ≤ M.
                   (If none in [i..n-1], return False immediately.)
                 * Once found, we have one valid segment [i..j].  count_segments += 1
                 * i = j+1
             - Finally, check if count_segments >= n-k. return True or False accordingly.

        5) Do a binary search on M in [0..hi].  Return the smallest M where feasible(M)=True.

        This yields the desired minimum possible final OR.

        NOTE: The overall idea (partition into contiguous segments, each AND ≤ M) works
              because that is exactly the condition needed so that the OR of segment ANDs
              is ≤ M.

        Let's implement it.
        """

        import math

        # ----------------------------
        # Build sparse table for AND
        # ----------------------------
        n = len(nums)
        LOG = int(math.log2(n)) + 1  # up to ~ 17 if n=1e5
        st = [[0]*n for _ in range(LOG)]
        # st[p][i] will store AND of the subarray from i to i+2^p-1, inclusive.

        # Initialize level-0 of sparse table
        for i in range(n):
            st[0][i] = nums[i]

        # Build it up
        j = 1
        length = 1
        while (length << 1) <= n:
            length <<= 1
            for i in range(n - length + 1):
                st[j][i] = st[j-1][i] & st[j-1][i + (length >> 1)]
            j += 1

        # Function to get AND of [L..R] in O(1)
        def get_and(L, R):
            length = R - L + 1
            p = length.bit_length() - 1  # floor(log2(length))
            return st[p][L] & st[p][R - (1 << p) + 1]

        # Precompute the overall OR (upper bound for our binary search)
        hi = 0
        for x in nums:
            hi |= x

        # Check feasibility: can we form >= (n-k) segments s.t. each segment-AND <= M?
        def feasible(M):
            seg_count = 0
            i = 0
            while i < n:
                # If even [i..n-1] has AND > M, no valid segment from i
                if get_and(i, n-1) > M:
                    return False
                # Otherwise binary-search for the minimal j in [i..n-1] with AND(i..j) <= M
                left = i
                right = n-1
                loc = right  # we know at least at right the AND might be <= M or we check inside
                while left <= right:
                    mid = (left + right)//2
                    if get_and(i, mid) <= M:
                        loc = mid
                        right = mid - 1
                    else:
                        left = mid + 1
                # loc is the smallest j with AND(i..j) <= M
                seg_count += 1
                i = loc + 1  # move to the next segment
            return (seg_count >= n - k)

        # ----------------------------
        # Binary search on the answer
        # ----------------------------
        lo = 0
        # hi is the OR of all elements
        ans = hi
        while lo <= hi:
            mid = (lo + hi) // 2
            if feasible(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1

        return ans