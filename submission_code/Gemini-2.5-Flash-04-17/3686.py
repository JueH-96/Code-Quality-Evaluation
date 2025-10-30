from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3: # Cannot split into three non-empty subarrays
            return 0

        # Use two different hash functions to reduce collision probability
        base1 = 31
        mod1 = 10**9 + 7
        base2 = 37
        mod2 = 10**9 + 9

        # Precompute powers of bases
        pows1 = [1] * (n + 1)
        pows2 = [1] * (n + 1)
        for k in range(1, n + 1):
            pows1[k] = (pows1[k-1] * base1) % mod1
            pows2[k] = (pows2[k-1] * base2) % mod2

        # Precompute prefix hashes
        # H[k] = hash of nums[0:k] using sum(nums[m] * base^(k-1-m)) mod MOD
        # H[0] = 0
        # H[k] = (H[k-1] * base + nums[k-1]) mod MOD for k >= 1
        H1 = [0] * (n + 1)
        H2 = [0] * (n + 1)
        for k in range(1, n + 1):
            H1[k] = (H1[k-1] * base1 + nums[k-1]) % mod1
            H2[k] = (H2[k-1] * base2 + nums[k-1]) % mod2

        # Helper function to get hash of subarray nums[a:b] (exclusive end index b)
        # Using hash definition H[k] = sum(nums[m] * base^(k-1-m))
        # hash(nums[a:b]) = H[b] - base^(b-a) * H[a]
        def get_hash(h_arr: List[int], pows_arr: List[int], mod: int, a: int, b: int) -> int:
            length = b - a
            if length <= 0:
                return 0
            # Calculate hash: H[b] - H[a] * base^length mod mod
            # Ensure result is non-negative
            hash_val = (h_arr[b] - h_arr[a] * pows_arr[length]) % mod
            return (hash_val + mod) % mod

        count = 0
        # Iterate through all possible split points i and j
        # nums1 = nums[0:i], nums2 = nums[i:j], nums3 = nums[j:n]
        # Requires 1 <= i < j < n
        # i is the end index of nums1 (exclusive), runs from 1 to n-2
        # j is the end index of nums2 (exclusive), runs from i+1 to n-1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # nums1 = nums[0:i], len1 = i
                # nums2 = nums[i:j], len2 = j - i
                # nums3 = nums[j:n], len3 = n - j

                is_beautiful = False

                # Condition 1: nums1 is a prefix of nums2
                len1 = i
                len2 = j - i
                if len1 <= len2:
                    # Compare nums[0:len1] with nums[i: i + len1]
                    # The segment nums[i: i + len1] is a prefix of nums2
                    # The end index for the second slice is i + len1.
                    # Since len1 <= len2 = j-i, we have i <= j-i, so 2*i <= j.
                    # Since j < n, 2*i < n. The end index is 2*i, which is <= n-1. Valid.
                    hash1_nums1 = get_hash(H1, pows1, mod1, 0, len1)
                    hash2_nums1 = get_hash(H2, pows2, mod2, 0, len1)
                    hash1_prefix_nums2 = get_hash(H1, pows1, mod1, i, i + len1)
                    hash2_prefix_nums2 = get_hash(H2, pows2, mod2, i, i + len1)

                    if hash1_nums1 == hash1_prefix_nums2 and hash2_nums1 == hash2_prefix_nums2:
                         is_beautiful = True

                # Condition 2: nums2 is a prefix of nums3
                len2 = j - i
                len3 = n - j
                if not is_beautiful and len2 <= len3: # Check only if not already beautiful
                    # Compare nums[i:j] with nums[j: j + len2]
                    # The segment nums[j: j + len2] is a prefix of nums3
                    # The end index for the second slice is j + len2.
                    # Since len2 <= len3 = n-j, we have j-i <= n-j, so 2*j - i <= n.
                    # The end index is 2*j - i, which is <= n. Valid.
                    hash1_nums2 = get_hash(H1, pows1, mod1, i, j)
                    hash2_nums2 = get_hash(H2, pows2, mod2, i, j)
                    hash1_prefix_nums3 = get_hash(H1, pows1, mod1, j, j + len2)
                    hash2_prefix_nums3 = get_hash(H2, pows2, mod2, j, j + len2)

                    if hash1_nums2 == hash1_prefix_nums3 and hash2_nums2 == hash2_prefix_nums3:
                         is_beautiful = True

                if is_beautiful:
                    count += 1

        return count