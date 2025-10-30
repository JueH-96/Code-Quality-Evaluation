class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        MODULUS = 10**9 + 7
        BASE = 31
        n = len(nums)
        
        # Precompute prefix hash array
        prefix_hash = [0] * (n + 1)
        power = [1] * (n + 1)
        for i in range(n):
            prefix_hash[i + 1] = (prefix_hash[i] * BASE + nums[i]) % MODULUS
            power[i + 1] = (power[i] * BASE) % MODULUS
        
        count = 0
        # Iterate over all possible i and j
        for i in range(1, n - 1):
            hash_nums1 = prefix_hash[i]
            for j in range(i + 1, n):
                hash_nums2 = (prefix_hash[j] - prefix_hash[i] * power[j - i]) % MODULUS
                hash_nums3 = (prefix_hash[n] - prefix_hash[j] * power[n - j]) % MODULUS
                
                # Check if nums1 is a prefix of nums2
                if j - i >= i:
                    hash_nums1_prefix = (prefix_hash[i + i] - prefix_hash[i] * power[i]) % MODULUS if i + i <= n else -1
                    if hash_nums1 == hash_nums1_prefix:
                        count += 1
                        continue  # No need to check the second condition
                
                # Check if nums2 is a prefix of nums3
                len_nums2 = j - i
                if n - j >= len_nums2:
                    hash_nums2_prefix = (prefix_hash[j + len_nums2] - prefix_hash[j] * power[len_nums2]) % MODULUS if j + len_nums2 <= n else -1
                    if hash_nums2 == hash_nums2_prefix:
                        count += 1
        return count