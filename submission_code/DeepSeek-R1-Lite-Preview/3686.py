class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        from typing import List
        
        N = len(nums)
        if N < 3:
            return 0  # At least three non-empty subarrays are needed
        
        # Parameters for rolling hash
        base = 911
        mod = 10**18 + 3  # A large prime to minimize collisions
        
        # Precompute powers of base
        power = [1] * (N + 1)
        for i in range(1, N + 1):
            power[i] = (power[i - 1] * base) % mod
        
        # Compute prefix hashes
        hash_prefix = [0] * (N + 1)
        for i in range(N):
            hash_prefix[i + 1] = (hash_prefix[i] * base + nums[i]) % mod
        
        def get_hash(a, b):
            # Returns the hash of nums[a:b]
            return (hash_prefix[b] - hash_prefix[a] * power[b - a]) % mod
        
        count = 0
        # Iterate over k from 1 to N-1
        for k in range(1, N):
            # Condition a: nums1 is a prefix of nums2
            if 2 * k <= N:
                hash_nums1 = get_hash(0, k)
                hash_nums2_prefix = get_hash(k, 2 * k)
                if hash_nums1 == hash_nums2_prefix:
                    count += (N - 2 * k)
            # Condition b: nums2 is a prefix of nums3
            # m should be from k+1 to floor((N + k)/2)
            m_max = (N + k) // 2
            for m in range(k + 1, m_max + 1):
                len2 = m - k
                len3 = N - m
                if len3 >= len2:
                    hash_nums2 = get_hash(k, m)
                    hash_nums3_prefix = get_hash(m, m + len2)
                    if hash_nums2 == hash_nums3_prefix:
                        count += 1
        return count