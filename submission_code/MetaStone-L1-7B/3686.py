class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        base = 911382629
        mod = 10**18 + 3
        
        # Precompute prefix hashes and power array
        prefix_hash = [0] * (n + 1)
        power = [1] * (n + 1)
        for i in range(n):
            prefix_hash[i + 1] = (prefix_hash[i] * base + nums[i]) % mod
            power[i + 1] = (power[i] * base) % mod
        
        count = 0
        
        for i in range(1, n - 1):
            # Case 1: j >= i, j <= n - i - 1
            j_case1_min = i
            j_case1_max = n - i - 1
            if j_case1_min > j_case1_max:
                continue
            for j in range(j_case1_min, j_case1_max + 1):
                a1, b1 = 0, i - 1
                hash1 = (prefix_hash[b1 + 1] - prefix_hash[a1] * power[b1 - a1 + 1]) % mod
                a2, b2 = i, i + j - 1
                hash2 = (prefix_hash[b2 + 1] - prefix_hash[a2] * power[b2 - a2 + 1]) % mod
                if hash1 == hash2:
                    count += 1
            
            # Case 2: j <= (n - i) // 2 and j <= n - i - 1
            j_max_case2 = min((n - i) // 2, n - i - 1)
            if j_max_case2 < 1:
                continue
            for j in range(1, j_max_case2 + 1):
                a2, b2 = i, i + j - 1
                hash2 = (prefix_hash[b2 + 1] - prefix_hash[a2] * power[b2 - a2 + 1]) % mod
                a3, b3 = i + j, i + j + j - 1
                hash3 = (prefix_hash[b3 + 1] - prefix_hash[a3] * power[b3 - a3 + 1]) % mod
                if hash2 == hash3:
                    count += 1
        
        return count