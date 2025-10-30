class Solution:
    def beautifulSplits(self, nums):
        n = len(nums)
        if n < 3:
            return 0
        
        mod = 10**18 + 3
        base = 10**9 + 7
        
        h = [0] * (n + 1)
        p_pows = [1] * (n + 1)
        
        for i in range(1, n + 1):
            h[i] = (h[i-1] * base + nums[i-1]) % mod
            p_pows[i] = (p_pows[i-1] * base) % mod
        
        count = 0
        
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                valid = False
                
                # Condition 1: nums1 is a prefix of nums2
                if j >= 2 * i and 2 * i <= n - 1:
                    hash1 = (h[i] - h[0] * p_pows[i]) % mod
                    hash2 = (h[2 * i] - h[i] * p_pows[i]) % mod
                    if hash1 == hash2:
                        valid = True
                
                # Condition 2: nums2 is a prefix of nums3
                if not valid and j <= (n + i) // 2:
                    len2 = j - i
                    end = j + len2
                    hash2_val = (h[j] - h[i] * p_pows[len2]) % mod
                    hash3_val = (h[end] - h[j] * p_pows[len2]) % mod
                    if hash2_val == hash3_val:
                        valid = True
                
                if valid:
                    count += 1
        
        return count