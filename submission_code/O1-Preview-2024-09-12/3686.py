class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        mod1 = 10**9 + 7
        mod2 = 10**9 + 9
        base1 = 911
        base2 = 1597

        # Precompute prefix hashes and base powers
        prefix_hash1 = [0] * (n + 1)
        prefix_hash2 = [0] * (n + 1)
        pow_base1 = [1] * (n + 1)
        pow_base2 = [1] * (n + 1)

        for i in range(1, n + 1):
            prefix_hash1[i] = (prefix_hash1[i - 1] * base1 + nums[i - 1]) % mod1
            prefix_hash2[i] = (prefix_hash2[i - 1] * base2 + nums[i - 1]) % mod2
            pow_base1[i] = (pow_base1[i - 1] * base1) % mod1
            pow_base2[i] = (pow_base2[i - 1] * base2) % mod2

        def get_hash(l, r):
            """Get hash of nums[l:r]"""
            hash1 = (prefix_hash1[r] - prefix_hash1[l] * pow_base1[r - l]) % mod1
            hash2 = (prefix_hash2[r] - prefix_hash2[l] * pow_base2[r - l]) % mod2
            return (hash1, hash2)

        count = 0
        for i in range(1, n - 1):
            len_nums1 = i
            # Check for nums1 is prefix of nums2
            max_j = n - 1
            for j in range(2 * i, n):
                if j - i < len_nums1:
                    continue  # len(nums2) < len(nums1), skip
                if i + len_nums1 > j:
                    continue  # nums2[0:len_nums1] exceeds nums2 length
                # Compare nums1 and nums2[0:len_nums1]
                hash1 = get_hash(0, i)
                hash2 = get_hash(i, i + len_nums1)
                if hash1 == hash2:
                    count += 1

            # Check for nums2 is prefix of nums3
            max_j = min(n - 1, (n + i) // 2)
            for j in range(i + 1, max_j + 1):
                len_nums2 = j - i
                if n - j < len_nums2:
                    continue  # len(nums3) < len(nums2), skip
                if j + len_nums2 > n:
                    continue  # nums3[0:len_nums2] exceeds nums3 length
                # Compare nums2 and nums3[0:len_nums2]
                hash1 = get_hash(i, j)
                hash2 = get_hash(j, j + len_nums2)
                if hash1 == hash2:
                    count += 1
        return count