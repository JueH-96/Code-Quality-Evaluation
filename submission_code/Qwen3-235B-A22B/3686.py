class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        mod = 10**9 + 7
        base = 262139  # A prime base for hashing
        
        h = [0] * (n + 1)
        pow_base = [1] * (n + 1)
        
        # Precompute prefix hashes and powers of the base
        for i in range(n):
            h[i + 1] = (h[i] * base + nums[i]) % mod
            pow_base[i + 1] = (pow_base[i] * base) % mod
        
        # Function to compute hash of nums[l:r]
        def get_hash(l, r):
            length = r - l
            res = (h[r] - h[l] * pow_base[length]) % mod
            return res if res >= 0 else res + mod
        
        count = 0
        
        # Iterate through all possible split points i and j
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                cond1 = False
                # Check Condition 1: nums1 is prefix of nums2
                if 2 * i <= j:
                    hash1 = get_hash(0, i)
                    hash2 = get_hash(i, 2 * i)
                    if hash1 == hash2:
                        cond1 = True
                
                cond2 = False
                # Check Condition 2: nums2 is prefix of nums3
                len2 = j - i
                end = j + len2
                if end <= n:
                    hash1_cond2 = get_hash(i, j)
                    hash2_cond2 = get_hash(j, end)
                    if hash1_cond2 == hash2_cond2:
                        cond2 = True
                
                if cond1 or cond2:
                    count += 1
        
        return count