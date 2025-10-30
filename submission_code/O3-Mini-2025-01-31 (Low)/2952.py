from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        total_initial = sum(nums1)
        total_rate = sum(nums2)
        
        # If no rate then the array never increases.
        if total_rate == 0:
            if total_initial <= x:
                return 0
            # we can remove one element per second:
            # At each second we can set one element to 0 until sum <= x.
            # Remove largest nums1 first.
            sorted_nums1 = sorted(nums1, reverse=True)
            curr = total_initial
            T = 0
            for val in sorted_nums1:
                T += 1
                curr -= val
                if curr <= x:
                    return T
            return -1

        # sort indices by nums2 descending
        combined = list(zip(nums1, nums2))
        combined.sort(key=lambda v: v[1], reverse=True)
        nums1_sorted = [v[0] for v in combined]
        nums2_sorted = [v[1] for v in combined]
        
        # Precompute prefix sums P1 and weighted prefix sums P2
        # P1[t] = sum_{j=0}^{t-1} nums1_sorted[j]
        # P2[t] = sum_{j=0}^{t-1} (j+1)*nums2_sorted[j]
        P1 = [0]*(n+1)
        P2 = [0]*(n+1)
        for i in range(n):
            P1[i+1] = P1[i] + nums1_sorted[i]
            P2[i+1] = P2[i] + (i+1)*nums2_sorted[i]
        
        # When T >= n, the optimal is to reset all indices.
        # In that case, final sum becomes: 
        # final_min = total_initial + T*total_rate - (total_initial + T*total_rate - n*total_rate + P2[n])
        #             = n*total_rate - P2[n]
        final_min = n*total_rate - P2[n]
        if final_min > x:
            return -1

        # Binary search T in [0, n] (T seconds)
        # Note: for T > n, final sum is the same as for T = n.
        lo, hi = 0, n
        ans = None
        while lo <= hi:
            mid = (lo + hi) // 2
            # if mid resets available (mid <= n) then we use exactly mid resets with factors (j+1)
            # final sum becomes:
            # final = total_initial + mid*total_rate - (P1[mid] + P2[mid])
            final = total_initial + mid*total_rate - (P1[mid] + P2[mid])
            if final <= x:
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        return ans if ans is not None else -1