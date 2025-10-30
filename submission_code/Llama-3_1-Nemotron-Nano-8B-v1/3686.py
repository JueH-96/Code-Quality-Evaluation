from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        base = 10**9 + 7
        mod = 10**18 + 3
        
        # Precompute prefix hashes and power array
        prefix_hash = [0] * (n + 1)
        power = [1] * (n + 1)
        for i in range(n):
            prefix_hash[i+1] = (prefix_hash[i] * base + nums[i]) % mod
            power[i+1] = (power[i] * base) % mod
        
        def get_hash(a: int, b: int) -> int:
            # Returns the hash of the subarray nums[a..b] (inclusive)
            if a > b:
                return 0
            length = b - a + 1
            hash_val = (prefix_hash[b+1] - (prefix_hash[a] * power[length]) % mod) % mod
            return hash_val
        
        count = 0
        
        # Iterate over all possible splits into three non-empty parts
        for i in range(1, n-1):
            for j in range(i+1, n):
                # Check condition a: nums1 is a prefix of nums2
                cond_a = False
                if j >= 2 * i and (i + i) <= n:
                    hash1 = get_hash(0, i-1)
                    hash2 = get_hash(i, i + i - 1)
                    cond_a = (hash1 == hash2)
                
                # Check condition b: nums2 is a prefix of nums3
                cond_b = False
                len_num2 = j - i
                if (n - j) >= len_num2:
                    hash2 = get_hash(i, j-1)
                    hash3 = get_hash(j, j + len_num2 - 1)
                    cond_b = (hash2 == hash3)
                
                if cond_a or cond_b:
                    count += 1
        
        return count