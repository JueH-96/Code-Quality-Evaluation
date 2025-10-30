class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        """
        We want the earliest second T so that after T "tick+operation" steps, 
        the sum(nums1) <= x. Each second we:
          1) add nums2[i] to nums1[i] for all i
          2) choose exactly one index i and reset nums1[i] to 0

        However, picking the same index more than once is never better than
        picking it exactly once at a later time (because the final state
        only cares about the value at the last pick). Consequently, we never
        need to pick any index more than once. Also, picking an index i at
        time t removes (nums1[i] + t*nums2[i]) from the final sum, but then
        from t+1,...,T it adds (T - t)*nums2[i]. Net effect (compared to not
        picking i at all) is removing nums1[i] + t*nums2[i]. Hence in the
        final sum, each index i contributes either
             nums1[i] + T*nums2[i]        (if not picked)
          or (T - t)*nums2[i]            (if picked exactly once at time t).

        The best way to use T picks is to pick up to T distinct indices—
        because re-picking the same index yields no extra reduction—and to
        schedule each picked index i at some time t in [1..T].

        If T ≥ n, we could pick all n indices (once each). If T < n, we can
        pick only T of them.

        To maximize the total removed, notice that removing index i at time t
        is nums1[i] + t*nums2[i]. We want to assign the largest times to
        the indices with the largest "growth rate" nums2[i]. If there's a tie
        in nums2[i], bigger nums1[i] also helps.

        So for each candidate T (0..n):
          • baseline = sum(nums1) + T * sum(nums2)
            (this is the sum if we never zero anything out over T steps).
          • pick r = min(T, n) indices with largest (nums2, nums1), and
            assign them times T, T-1, ..., T - (r-1).
          • compute the sum of removed = Σ [ nums1[i_k] + (assigned_time)*nums2[i_k] ].
          • final_sum = baseline - sum_removed.
          • check if final_sum <= x.

        We return the smallest T for which final_sum <= x, or -1 if none works.
        """

        import math

        n = len(nums1)
        sum1 = sum(nums1)
        sum2 = sum(nums2)

        # Pair up (nums2[i], nums1[i], i) so we can sort primarily by nums2 desc, secondarily by nums1 desc
        pairs = sorted(zip(nums2, nums1), key=lambda p: (p[0], p[1]), reverse=True)

        # Try all T from 0 to n
        for T in range(n+1):
            # Baseline if we never pick any element
            baseline = sum1 + T * sum2

            # We can pick at most T indices (if T <= n) or n indices (if T>n)
            r = min(T, n)

            # Greedily remove the largest possible amounts by assigning
            # the largest times T, T-1, ..., T-r+1 to the r largest pairs
            total_removed = 0
            for k in range(r):
                g2, g1 = pairs[k]  # nums2, nums1
                # assign this index the time = T - k (descending times)
                time_slot = T - k
                total_removed += g1 + time_slot*g2

            final_sum = baseline - total_removed

            if final_sum <= x:
                return T

        # If we never succeed
        return -1