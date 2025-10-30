class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        """
        We want to apply up to k "merge" operations on adjacent elements, where merging
        nums[i] and nums[i+1] replaces them by (nums[i] & nums[i+1]).  After these merges,
        we get some final array, and we want to minimize the bitwise OR of its elements.

        ----------------------------------------------------------------------
        Key Reformulation:
          • Each merge of adjacent elements is effectively forming a contiguous "segment"
            whose final value is the AND of all elements in that segment.
          • After up to k merges on an array of length n, we end up with between (n-k) and n
            segments (because each merge reduces the number of segments by 1).
          • The final bitwise OR is then the OR of the AND-values of these segments.

          So the question becomes:  can we split the array into s segments, each segment's
          AND ≤ X, for some s in the range [n-k, n]?  If yes, then the final OR ≤ X.

          Equivalently, define s_max(X) = the maximum number of contiguous segments into which
          we can partition nums so that each segment's AND is ≤ X.  Then the number of merges
          needed for that partition is n - s_max(X).  If n - s_max(X) ≤ k (i.e. s_max(X) ≥ n - k),
          we can achieve a final OR ≤ X.

        Hence the core subproblem is:
           "Given X, compute s_max(X)."

        ----------------------------------------------------------------------
        Computing s_max(X):

        We need a way to find the maximum number of contiguous subarrays (segments) each having
        bitwise-AND ≤ X.  A straightforward dynamic-programming definition is:

            dp[i] = the maximum number of valid segments covering nums[:i+1].

        Then dp[i] = max( dp[j-1] + 1 ) over all j ≤ i,
        subject to AND of nums[j..i] ≤ X;  and dp[-1] = 0 as a base case.

        Naively, checking all j would be O(n^2).  However, there is a known optimization
        for "subarray-GCD" or "subarray-min" partitioning problems that also carries over
        to "subarray-bitwise-AND," because bitwise AND can drop bits only so many times.

        The trick is:
          • We scan from left to right.
          • Maintain a small "stack" (or list) of pairs (val, segCount), where 'val' is a possible
            intersection value of a subarray ending exactly at the previous position, and 'segCount'
            is the best number of segments we can achieve up to that subarray boundary.
          • When we move to a new element a[i], we build a fresh "new list" of intersection values
            for subarrays that end exactly at i.
          • We combine a[i] with each old intersection (from right to left) by ANDing them.  Because
            AND only loses bits, we get at most ~30 distinct intermediate intersection values (for 30-bit numbers).
          • Each time we get a new intersection ≤ X, we can form a new segment, so we update
            the candidate dp value accordingly.

        In the end, dp will give us s_max(X) = the maximum number of segments partitioning
        nums[0..i].  After processing i = n-1, dp is the maximum count of segments in the entire array.

        Once we can compute s_max(X) in O(n * 30), we do a binary search over X from 0..(OR of all nums).
        The final answer is the smallest X for which s_max(X) ≥ (n - k).  That X is the minimum possible
        bitwise-OR we can achieve after up to k merges.

        ----------------------------------------------------------------------
        Complexity:
          • Let N = len(nums), and B ~ 30 (since nums[i] < 2^30).
          • Building s_max(X) takes O(N*B) time.
          • Binary searching X over at most ~31 possible ranges (0..maxOR) takes O(B * N * B) = O(B^2 * N),
            which is around 30*30*N = 900*N.  For N up to 1e5 this is borderline, but with efficient
            implementation (and in faster languages) it can pass.  In Python, one needs to be mindful
            of implementation details.

        Below is an implementation of this approach.
        """

        import sys
        sys.setrecursionlimit(10**7)

        # ------------------------------------------------------------
        # Precompute the overall OR to bound our binary-search range.
        # ------------------------------------------------------------
        overall_or = 0
        for v in nums:
            overall_or |= v

        # ------------------------------------------------------------
        # Function that computes s_max(x): the maximum number of
        # contiguous segments whose AND ≤ x.
        #
        # We'll implement the "stack of intersection" trick from left
        # to right.  Let dp be the maximum # of segments so far.
        # We'll keep a list of (val, segCount), meaning: "There is a
        # subarray ending at the previous index with bitwise-AND = val,
        # and up to that boundary we can form segCount segments in total."
        #
        # Then for a new element a[i], we build a new list new_st by
        # combining a[i] with each old intersection from right to left,
        # but we keep only distinct values for val.
        #
        # The best #segments in the new list for val ≤ x will be seg+1,
        # because we can form a new segment from that boundary up to i.
        #
        # dp[i] itself will be the maximum segment count among new_st.
        # We'll keep track globally in "dp" variable as we go.
        # ------------------------------------------------------------
        def compute_smax(x: int) -> int:
            # dp_so_far = the maximum segments we've formed scanning up to just before this i.
            dp_so_far = 0
            # st will hold distinct (val, segCount) for subarrays that end exactly at i-1.
            # Start empty (meaning no subarray yet, dp=0).
            st = []

            for a in nums:
                # We'll build new_st for subarrays that end exactly at "this" index i.
                new_st = []
                # Start by forming a possible new segment with only the element a itself
                # (if it's <= x, that increment in segments is dp_so_far + 1, else we can't).
                if a <= x:
                    best_here = dp_so_far + 1
                else:
                    best_here = dp_so_far  # no increment if a alone is > x
                new_val = a

                # Now combine from right-to-left with pairs in st
                # st holds pairs (old_val, old_dp).  old_val is the AND
                # of some subarray that ended exactly at the previous index,
                # and old_dp is how many segments we formed up to that boundary.
                #
                # We'll go in reverse order so that we effectively handle smaller sub-sub arrays first.
                # But an optimization is to go from last to first, so we just do reversed(st).
                # Because AND can only lose bits, each time we do new_val = old_val & a,
                # the new_val might get smaller and possibly go below x, letting us form a new segment.
                #
                # We'll keep track of distinct new_val in new_st.
                for (old_val, old_dp) in reversed(st):
                    comb = old_val & a
                    if comb == new_val:
                        # same intersection, just update the best segment count if comb <= x
                        if comb <= x:
                            # we can form one more segment from that old boundary
                            best_here = max(best_here, old_dp + 1)
                        # no need to add more pairs since the intersection didn't change
                        # we break because continuing won't produce new intersection values
                        break
                    else:
                        # intersection changed
                        new_val = comb
                        if new_val <= x:
                            best_here = max(best_here, old_dp + 1)
                        # see if we can merge it into the last element in new_st
                        if not new_st or new_st[-1][0] != new_val:
                            new_st.append((new_val, old_dp))
                        # if it becomes 0, further ANDs won't change, so we can break
                        if new_val == 0:
                            break

                # Now we have a candidate best_here which is the best #segments we can get
                # ending at this index.  We'll maintain dp_so_far as the global maximum so far.
                dp_so_far = max(dp_so_far, best_here)

                # Also we push the single-element pair (a, dp_so_far) onto new_st,
                # but first we see if combining it with the last new_val is the same or not.
                # Actually we started new_st with possible combos from st.  We also want
                # to store (a, dp_so_far) if it differs from the last pair's val.
                if new_st and new_st[-1][0] == a:
                    # if the top has the same val, we only increase the dp if dp_so_far is bigger
                    new_st[-1] = (a, max(new_st[-1][1], dp_so_far))
                else:
                    new_st.append((a, dp_so_far))

                # Now new_st is the stack for the next iteration.
                st = new_st

            # dp_so_far after the last element is the maximum # of segments we formed in total.
            return dp_so_far

        # ------------------------------------------------------------
        # Define a function feasible(mid): returns True if we can
        # achieve final OR ≤ mid with at most k merges.  That is
        # s_max(mid) ≥ (n - k).
        # ------------------------------------------------------------
        def feasible(mid: int) -> bool:
            smax_val = compute_smax(mid)
            return (smax_val >= len(nums) - k)

        # ------------------------------------------------------------
        # Binary-search over the possible final OR in range [0..overall_or].
        # We'll do a standard integer binary-search for the minimal value
        # that is feasible.
        # ------------------------------------------------------------
        low, high = 0, overall_or
        ans = overall_or

        while low <= high:
            mid = (low + high) // 2
            if feasible(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1

        return ans