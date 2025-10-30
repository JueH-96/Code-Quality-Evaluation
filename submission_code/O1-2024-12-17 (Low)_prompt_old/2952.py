class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        """
        We want the earliest time T such that after T "increments" (each nums1[i] += nums2[i])
        and T resets in total (one reset per second, possibly on the same or different indices),
        the sum of nums1 is <= x.

        A key insight is that allowing multiple resets on the same index i can (in total) remove
        up to (nums1[i] + T * nums2[i]) from index i’s contribution—i.e. we can effectively
        wipe out i's entire final value if we reset it T times.  However, we have only T resets
        total (distributed among all indices), so the question is how best to allocate these T resets.

        ---------------------------------------------------------------------

        1) Let n = len(nums1), and define:
             a_i = nums1[i]
             b_i = nums2[i]
           Then if we did no resets at all, at time T each element i is a_i + T*b_i,
           so the sum would be S_no_resets(T) = sum(a_i) + T * sum(b_i).

        2) Each reset at second t on index i effectively "removes" some portion of i's growth.
           In fact, if we choose to reset index i k times (out of T possible resets), we can
           remove exactly (a_i + k * b_i) from i's final contribution compared to no resets.
           (Because after k resets spread out in time, i’s final value is (T - k)*b_i, so we
            reduce from (a_i + T*b_i) down to (T - k)*b_i, a net removal of a_i + k*b_i.)

        3) Therefore, the maximum total removal we can get with T resets is the best way to
           distribute those T resets among all n indices.  Each index i can use from 0 up to T
           resets, removing (a_i + k*b_i).  Summing k across all i must be T.

           Equivalently, we can think of each "reset" as a "pick" from a multiset of values:
             • The first time we pick index i, it contributes (a_i + b_i).
             • Each subsequent time (2nd, 3rd, ...) we pick index i, it contributes b_i.
           We want exactly T picks (since T resets total).

        4) However, enumerating all distributions can be expensive for large T.  But notice
           we only ever want to pick from whichever b_i is largest or from a first-pick
           whose (a_i + b_i) is large.  A more direct “greedy” approach works:

           - Let Bmax = max(b_i).  For the "subsequent picks" after the first reset of a given
             item, each pick is worth b_i.  Among all b_i, the best subsequent pick is always
             the largest b_i.  Once we have used a "first pick" on at least one item i that has
             b_i = Bmax, we can get Bmax for any number of subsequent picks.  
           
           - Meanwhile, a "first pick" from index i is worth (a_i + b_i).  We only want to use
             that if (a_i + b_i) is bigger than or equal to picking Bmax instead.

           Therefore, for T picks:
             • Sort the array firstPicks = [a_i + b_i for i in range(n)] in descending order.
             • Let c = number of values in firstPicks that are > Bmax.  (Those are definitely
               better than just picking Bmax.)
             • If T <= c, we can take exactly the top T of those big first-picks.
             • If T > c, we take all those c large first-picks and then fill the remaining (T-c)
               picks with Bmax each (since once we’ve “unlocked” Bmax from an item that actually
               has b_i = Bmax, or if we prefer, we consider Bmax as the best subsequent pick).

           So, the maximum removal with T resets is:
               removal(T) = sum of top min(T, c) elements of firstPicks
                            + ( T - min(T, c) ) * Bmax
                        =  prefixSumOfFirstPicks[min(T,c)] + (T - min(T,c)) * Bmax
           
           Then the final sum at time T is:
               finalSum(T) = (sum(a_i) + T * sum(b_i)) - removal(T).

        5) We do a (modified) binary search on T to find the smallest T for which finalSum(T) <= x.
           - If sum(nums1) <= x initially, answer is 0.
           - Otherwise, we keep doubling T (1,2,4,8,...) and checking finalSum(T) until either
             finalSum(T) <= x or we exceed some large cap (e.g. 2e7).  If we never get <= x,
             return -1.
           - Then we binary search the range where finalSum(T) crosses <= x.

        This runs efficiently because:
          - Sorting "firstPicks" is O(n log n).
          - We build a prefix sum array for those picks for quick sums.
          - Each check finalSum(T) is then O(1).
          - Doubling + binary searching up to ~10^7 or so takes ~log(10^7) steps, each O(1).

        ---------------------------------------------------------------------
        """

        import math

        n = len(nums1)
        sumA = sum(nums1)
        sumB = sum(nums2)

        # If already <= x, 0 time needed
        if sumA <= x:
            return 0

        # Prepare for the "maximum removal" computation
        # Bmax = max of b_i
        Bmax = max(nums2)

        # Build the firstPickValue array and sort descending
        # firstPickValue[i] = a_i + b_i
        firstPickValues = [a + b for a, b in zip(nums1, nums2)]
        firstPickValues.sort(reverse=True)

        # Build prefix sums, prefixSum[k] = sum of top k elements
        prefixSum = [0] * (n+1)
        for i in range(n):
            prefixSum[i+1] = prefixSum[i] + firstPickValues[i]

        # Count how many of these are strictly greater than Bmax
        # (Those we definitely prefer over picking Bmax alone.)
        # Actually ">= Bmax" could matter, but since a>=1, a+b > b, so
        # a+b > b_i always, so it's always >= b_i. We only need strictly > Bmax.
        c = 0
        for val in firstPickValues:
            if val > Bmax:
                c += 1
            else:
                break  # since it's sorted descending, we can stop when val <= Bmax

        # Function to compute finalSum(T) in O(1)
        def final_sum(T: int) -> int:
            # removal(T) = sum of top min(T,c) elements + (T - min(T,c)) * Bmax
            pick_first = min(T, c)
            removal = prefixSum[pick_first] + (T - pick_first) * Bmax
            return (sumA + T * sumB) - removal

        # Check a quick bound: if even resetting everything fully won't help:
        # The theoretical maximum removal is sum_i (a_i + T*b_i) if we put all T resets into each i,
        # but that would require T*n resets in total if we tried to remove everything. Not feasible
        # to check that directly. Instead, we rely on the doubling approach.

        # If Bmax == 0, then each second does not grow. The maximum T we might ever need is n,
        # because each reset can zero out one element. If sumA > x but n resets won't fix it,
        # answer is -1. We can handle that inside the same framework anyway.

        # If we can't find T with final_sum(T) <= x up to some large range, answer -1.
        # We'll do exponential (doubling) search up to a big limit, then binary search.
        # Reasonable big limit, e.g. 2*(10^7) (to be safe).
        # Each check is O(1), so this is still fast enough.

        # First, quickly check if T=10^7 or so obviously can't work => we give up and return -1.
        # But let's implement a systematic doubling search.

        # Edge case: if Bmax == 0, let's set a smaller upper limit = n
        # (since after n resets we can zero out all elements if needed).
        # Otherwise, we can go bigger.

        # 1) Doubling until final_sum(mid) <= x or mid > some large cap
        cap = n if Bmax == 0 else 2_000_000  # somewhat arbitrary big limit
        lo, hi = 0, 1
        # We know final_sum(0) = sumA > x, so 0 is not valid. We'll start hi=1.
        while True:
            if hi > cap:
                # We'll check final_sum(cap). If still > x, answer -1
                if final_sum(cap) > x:
                    return -1
                # otherwise we'll binary search in [lo, cap]
                hi = cap
                break

            if final_sum(hi) <= x:
                break
            lo = hi
            hi <<= 1  # double hi

        # 2) Now we have final_sum(lo) > x, final_sum(hi) <= x. Binary search in [lo+1, hi].
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if final_sum(mid) <= x:
                hi = mid
            else:
                lo = mid

        return hi