from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        """
        We want to split nums into three contiguous subarrays nums1, nums2, nums3 such that:
          1) nums = nums1 + nums2 + nums3  (concatenation in order), and
          2) Either nums1 is a prefix of nums2, OR nums2 is a prefix of nums3.

        Let n = len(nums). We require three non-empty parts, so we choose two cut indices i < j:
           nums1 = nums[0..i],  nums2 = nums[i+1..j],  nums3 = nums[j+1..n-1].

        "nums1 is a prefix of nums2" means:
           len(nums1) <= len(nums2)  and  nums1 == nums2[: len(nums1)].
           Concretely, if nums1 = nums[0..i] has length i+1,
           then we need j - (i+1) + 1 >= i+1  =>  j >= 2i+1,
           and nums[k] == nums[i+1 + k] for k=0..i.

        "nums2 is a prefix of nums3" means:
           len(nums2) <= len(nums3)  and  nums2 == nums3[: len(nums2)].
           If nums2 = nums[i+1..j] has length L = j - i,
           then we need n-1 - j >= L, or j <= (n-1 + i)//2,
           and nums[i+1..j] == nums[j+1.. j+L].

        We count all (i,j) that satisfy either condition. To avoid double-counting
        those that satisfy both conditions, we do:

          1) Pre-check "nums1 is prefix of nums2":
             • If nums[0..i] == nums[i+1..i+1+(i)] (i.e. the first i+1 elements of nums2),
               then for every j >= 2i+1 (up to n-2 so nums3 is non-empty), that split works.
          2) Check "nums2 is prefix of nums3" for other (i,j).
             • We test subarray equality with rolling-hash in O(1) per check,
               but only for those j that were not already counted by condition (1).

        Because n ≤ 5000, a carefully-implemented O(n^2) solution with rolling hash can pass.
        """

        n = len(nums)
        if n < 3:
            return 0

        # -----------------------------
        # 64-bit rolling-hash utilities (to reduce Python overhead, we use masking).
        # We will use two different bases for a double-hash approach to minimize collisions.
        # -----------------------------
        mask64 = (1 << 64) - 1

        def mul64(a, b):
            return (a * b) & mask64

        def add64(a, b):
            return (a + b) & mask64

        # Choose two bases (large and distinct).
        base1 = 131542391
        base2 = 137

        # Precompute powers and prefix-hashes for both hashes.
        H1_a = [0] * (n + 1)  # first hash array
        H1_b = [0] * (n + 1)  # second hash array
        P1_a = [1] * (n + 1)  # powers for first hash
        P1_b = [1] * (n + 1)  # powers for second hash

        for i in range(n):
            # First hash
            H1_a[i+1] = add64(mul64(H1_a[i], base1), nums[i] + 1)
            P1_a[i+1] = mul64(P1_a[i], base1)
            # Second hash
            H1_b[i+1] = add64(mul64(H1_b[i], base2), nums[i] + 1)
            P1_b[i+1] = mul64(P1_b[i], base2)

        def get_hash_pair(L, R):
            """
            Returns the pair of 64-bit rolling-hash values for subarray nums[L..R].
            """
            length = R - L + 1
            # First hash
            ha = (H1_a[R+1] - mul64(H1_a[L], P1_a[length])) & mask64
            # Second hash
            hb = (H1_b[R+1] - mul64(H1_b[L], P1_b[length])) & mask64
            return (ha, hb)

        # ------------------------------------------------
        # 1) Precompute which i satisfy (nums1 is a prefix of nums2).
        #
        #    If nums[0..i] == nums[i+1..2i+1], then for j in [2i+1..n-2]
        #    the pair (i,j) is valid for condition 1 (assuming subarray3 is non-empty).
        # ------------------------------------------------
        prefixMatch1 = [False] * n
        for i in range(n - 1):
            # We need 2i+1 < n to have a subarray of length i+1 at i+1..2i+1
            if 2*i + 1 < n:
                # Compare subarray [0..i] with [i+1..2i+1]
                if get_hash_pair(0, i) == get_hash_pair(i+1, 2*i+1):
                    prefixMatch1[i] = True

        # Count how many splits from condition 1
        count1 = 0
        for i in range(n - 2):
            if prefixMatch1[i]:
                start_j = 2*i + 1
                if start_j <= n - 2:
                    count1 += (n - 2) - start_j + 1
        # count1 is the total contributed by "nums1 prefix of nums2" condition

        # ------------------------------------------------
        # 2) Now handle "nums2 is prefix of nums3" for pairs (i,j) not already counted.
        #
        #    For i, let subarray2 = nums[i+1..j]. Length = j - i.
        #    We want subarray2 = first (j - i) elements of subarray3 = nums[j+1..].
        #    This requires j <= (n-1 + i)//2 and j < n-1 (for subarray3 non-empty).
        #
        #    We skip those (i,j) where prefixMatch1[i] is True and j >= 2i+1,
        #    because they are already counted under condition 1.
        # ------------------------------------------------
        count2 = 0
        for i in range(n - 2):
            # subarray2 length L = j - i.
            # j <= (n-1 + i)//2 => L <= (n-1+i)//2 - i.
            zmax = (n - 1 - i) // 2  # max length L for subarray2
            if zmax < 1:
                continue

            # If prefixMatch1[i] is True, then all j >= 2i+1 are counted by condition1,
            # meaning all L >= (2i+1 - i) = i+1 are already counted.
            # So we only need to check L < i+1 in that case.
            if prefixMatch1[i]:
                lmax = min(zmax, i)  # only up to i
            else:
                lmax = zmax

            for L in range(1, lmax + 1):
                # subarray2 = [i+1.. i+L]
                # subarray3 prefix to match = [i+L+1.. i+2L], each length L
                # Check if they match
                if get_hash_pair(i+1, i+L) == get_hash_pair(i+L+1, i+2*L):
                    count2 += 1

        return count1 + count2