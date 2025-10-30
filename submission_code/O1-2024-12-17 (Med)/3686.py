from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        """
        We split nums into (nums1, nums2, nums3) at positions i, j
          - nums1 = nums[0:i]
          - nums2 = nums[i:j]
          - nums3 = nums[j:]
        A "beautiful" split is one where either:
            1) nums1 is a prefix of nums2   OR
            2) nums2 is a prefix of nums3
        We want the total count of such splits.

        Let n = len(nums).

        1) If nums1 is a prefix of nums2, then for length i = len(nums1),
           we must have subarray(0..i-1) == subarray(i..2i-1)  (if 2i <= n).
           If that holds, then for j in [2i..n-1], that split satisfies condition1.
           So we can count these in O(n) if prefix-check is O(1).

        2) If nums2 is a prefix of nums3, then for the split at (i, j),
           we need subarray(i..j-1) == subarray(j..(j + (j-i) - 1)),
           and also length(nums2) <= length(nums3) => (j-i) <= (n-j).
           => j <= (n + i) // 2. We'll count for each i, all j in [i+1.. min(n-1, (n+i)//2]]
           that satisfy the prefix-check. Checking each takes O(1) with hashing,
           so in worst case O(n^2) overall.

        We sum #splits from condition1 and #splits from condition2,
        but subtract those that satisfy both (inclusion-exclusion).

        We'll use a double rolling hash to reduce collisions:
            h1, h2 with different moduli.
        """

        # 1) Build double rolling hash for nums
        #    prefix1[i] = hash of nums[:i] under mod1
        #    prefix2[i] = hash of nums[:i] under mod2
        #    We will use powers of a base for O(1) subarray-hash extraction.
        base  = 257
        mod1  = 10**9 + 7
        mod2  = 10**9 + 9
        n = len(nums)

        prefix1 = [0]*(n+1)
        prefix2 = [0]*(n+1)
        power1  = [1]*(n+1)
        power2  = [1]*(n+1)

        for i in range(n):
            prefix1[i+1] = (prefix1[i]*base + nums[i]) % mod1
            prefix2[i+1] = (prefix2[i]*base + nums[i]) % mod2

        for i in range(n):
            power1[i+1] = (power1[i]*base) % mod1
            power2[i+1] = (power2[i]*base) % mod2

        def get_hash(l: int, r: int):
            """
            Return the double-hash (h1, h2) for subarray nums[l..r].
            """
            length = r - l + 1
            h1 = prefix1[r+1] - (prefix1[l]*power1[length] % mod1)
            h2 = prefix2[r+1] - (prefix2[l]*power2[length] % mod2)
            return (h1 % mod1, h2 % mod2)

        def subeq(l1: int, l2: int, length: int) -> bool:
            """
            Check if subarray nums[l1..l1+length-1] == nums[l2..l2+length-1].
            Uses O(1) hashing lookups.
            """
            return get_hash(l1, l1+length-1) == get_hash(l2, l2+length-1)

        # 2) Count those satisfying condition1: nums1 is prefix of nums2
        #    We only check i where 2*i <= n, and subeq(0, i, i).
        #    For each such i, j can be [2i..n-1].
        condition1_count = 0
        for i in range(1, n // 2 + 1):
            if subeq(0, i, i):
                # j can be from 2*i to n-1
                condition1_count += (n - 2*i)

        # 3) Count those satisfying condition2: nums2 is prefix of nums3
        #    For each i in [1..n-1], j in [i+1..min(n-1, (n+i)//2]]
        condition2_count = 0
        for i in range(1, n):
            max_j = (n + i) // 2
            if max_j > n - 1:
                max_j = n - 1
            for j in range(i+1, max_j+1):
                # Check if subarray(i..j-1) is prefix of subarray(j..2j-i-1)
                if subeq(i, j, j - i):
                    condition2_count += 1

        # 4) Count those that satisfy both condition1 and condition2
        #    Means i, j must be valid for both:
        #         - j >= 2i and subeq(0,i,i)  (condition1)
        #         - j <= (n + i)//2 and subeq(i,j,j-i) (condition2)
        condition12_count = 0
        for i in range(1, n // 2 + 1):
            if not subeq(0, i, i):
                continue
            lower_j = 2*i
            if lower_j > n - 1:
                break
            upper_j = (n + i) // 2
            if upper_j > n - 1:
                upper_j = n - 1
            for j in range(lower_j, upper_j+1):
                if subeq(i, j, j - i):
                    condition12_count += 1

        return condition1_count + condition2_count - condition12_count