class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        base = 101
        mod = 10**9 + 7
        hash_prefix = [0] * (n + 1)
        power = [1] * (n + 1)
        
        # Precompute prefix hashes and powers
        for i in range(n):
            hash_prefix[i + 1] = (hash_prefix[i] * base + nums[i] + 1) % mod
            power[i + 1] = (power[i] * base) % mod
        
        # Function to get hash of subarray nums[l:r]
        def get_hash(l, r):
            return (hash_prefix[r] - hash_prefix[l] * power[r - l]) % mod
        
        res = 0
        
        # Condition A: nums1 is a prefix of nums2
        max_i = (n - 1) // 2
        for i in range(1, max_i + 1):
            if get_hash(0, i) == get_hash(i, 2 * i):
                res += n - 2 * i
        
        # Condition B: nums2 is a prefix of nums3
        for length in range(1, n // 2 + 1):
            for i in range(n - 2 * length + 1):
                if get_hash(i, i + length) == get_hash(i + length, i + 2 * length):
                    res += 1
        
        return res