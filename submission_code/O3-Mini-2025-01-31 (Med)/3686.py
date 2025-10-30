from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        # We'll use two independent mod values for rolling hash.
        mod1 = 10**9 + 7
        mod2 = 10**9 + 9
        base = 131
        
        # Precompute prefix hashes and powers for both mod values.
        H1 = [0] * (n + 1)
        H2 = [0] * (n + 1)
        P1 = [1] * (n + 1)
        P2 = [1] * (n + 1)
        for i in range(n):
            # Use (nums[i]+1) so that 0's do not cause problems.
            H1[i+1] = (H1[i] * base + (nums[i] + 1)) % mod1
            H2[i+1] = (H2[i] * base + (nums[i] + 1)) % mod2
            P1[i+1] = (P1[i] * base) % mod1
            P2[i+1] = (P2[i] * base) % mod2
        
        # Helper inline function to get hash for subarray nums[l:r]
        # We inline it inside our loops for speed.
        def get_hash(l, r):
            # compute hash for interval [l, r)
            len_seg = r - l
            a = H1[r] - H1[l] * P1[len_seg] % mod1
            b = H2[r] - H2[l] * P2[len_seg] % mod2
            if a < 0: a += mod1
            if b < 0: b += mod2
            return (a, b)
        
        res1 = 0  # count of splits satisfying condition1: nums1 is prefix of nums2.
        # For a given split defined by indices (i, j) with:
        #   nums1 = nums[0:i], nums2 = nums[i:j], nums3 = nums[j:].
        # Condition 1 holds if:
        #   len(nums1) <= len(nums2) (i <= j-i) and the first i elements of nums2 equal nums1.
        # Notice: If nums[0:i] == nums[i:2*i] then for any j >= 2*i (and j ≤ n-1 because nums3 non-empty)
        # the condition holds.
        # i can range from 1 to floor((n-1)/2) because j must be at least 2*i and j <= n-1.
        max_i = (n - 1) // 2  # ensures 2*i <= n-1.
        for i in range(1, max_i + 1):
            # Check if nums1 equals the prefix of nums2, i.e. compare nums[0:i] with nums[i:2*i]
            if get_hash(0, i) == get_hash(i, 2*i):
                # Then any j from 2*i up to n-1 gives a valid split
                res1 += (n - 2 * i)
        
        res2 = 0  # count of splits satisfying condition2: nums2 is prefix of nums3.
        # For condition 2, for a fixed j (the boundary between nums2 and nums3) with 2 <= j <= n-1,
        # we need to pick an i such that:
        #   1 <= i < j, and additionally the length condition: len(nums2)=j-i <= len(nums3)=n-j,
        #   i >= 2*j - n.
        # Then the condition requires that the subarray nums2, i.e. nums[i:j],
        # equals the first (j-i) elements of nums3, i.e. nums[j: j+(j-i)].
        # We can reparameterize by L = j-i where L >= 1 and L <= n-j.
        # Also, i = j - L must be at least max(1, 2*j-n).
        for j in range(2, n):  # j from 2 to n-1
            lower_i = 1 if (2 * j - n) < 1 else (2 * j - n)  # i must be >= max(1, 2*j-n)
            # i goes from lower_i to j-1. Let L = j - i.
            # So L runs from 1 to j - lower_i.
            maxL = j - lower_i
            # Also, we need L <= n - j (ensuring nums3 is at least as long as nums2)
            if maxL > (n - j):
                maxL = n - j
            # Now iterate L from 1 to maxL.
            # For each L, i becomes j - L.
            for L in range(1, maxL + 1):
                i_idx = j - L
                # Check if nums[i:j] equals nums[j: j+L]
                if get_hash(i_idx, j) == get_hash(j, j + L):
                    res2 += 1
                    
        # Now, count the splits that satisfy BOTH conditions. 
        # These splits satisfy:
        #  (a) Condition1: nums1 is prefix of nums2, i.e. get_hash(0,i)==get_hash(i,2*i) for subarray nums1 = nums[0:i]
        #  (b) Condition2: nums2 is prefix of nums3, i.e. for j with j >= 2*i and j - i <= n - j,
        #      and get_hash(i, j) == get_hash(j, j+(j-i)).
        res_both = 0
        # We already know that for condition1 we require i from 1 to floor((n-1)/2) and only if
        # get_hash(0,i)==get_hash(i,2*i) holds.
        for i in range(1, max_i + 1):
            if get_hash(0, i) != get_hash(i, 2*i):
                continue
            # For a fixed i, j can range from 2*i to n-1, but must also satisfy j-i <= n-j.
            # That inequality becomes: j <= (n + i) // 2.
            j_upper = (n + i) // 2
            for j in range(2 * i, j_upper + 1):
                L = j - i  # length of nums2
                # Check that j+L is within bounds (it must be <= n)
                if j + L > n:
                    continue
                if get_hash(i, j) == get_hash(j, j + L):
                    res_both += 1
        
        # The total number of beautiful splits is the union of splits satisfying condition1 or condition2.
        # Use inclusion-exclusion to avoid double counting splits that satisfy both.
        return res1 + res2 - res_both