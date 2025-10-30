class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)
        mod1 = 10**9 + 7
        mod2 = 10**9 + 9
        base1 = 131
        base2 = 137
        
        h1_s = [0] * (n + 1)
        h2_s = [0] * (n + 1)
        pow1 = [1] * (n + 1)
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            char_val = ord(s[i - 1]) - ord('a') + 1
            h1_s[i] = (h1_s[i - 1] * base1 + char_val) % mod1
            h2_s[i] = (h2_s[i - 1] * base2 + char_val) % mod2
            pow1[i] = (pow1[i - 1] * base1) % mod1
            pow2[i] = (pow2[i - 1] * base2) % mod2
        
        h1_p = [0] * (m + 1)
        h2_p = [0] * (m + 1)
        for i in range(1, m + 1):
            char_val = ord(pattern[i - 1]) - ord('a') + 1
            h1_p[i] = (h1_p[i - 1] * base1 + char_val) % mod1
            h2_p[i] = (h2_p[i - 1] * base2 + char_val) % mod2
        
        def get_hash_s(l, r):
            seg_len = r - l
            hash1 = (h1_s[r] - h1_s[l] * pow1[seg_len]) % mod1
            hash2 = (h2_s[r] - h2_s[l] * pow2[seg_len]) % mod2
            return (hash1, hash2)
        
        def get_hash_p(l, r):
            seg_len = r - l
            hash1 = (h1_p[r] - h1_p[l] * pow1[seg_len]) % mod1
            hash2 = (h2_p[r] - h2_p[l] * pow2[seg_len]) % mod2
            return (hash1, hash2)
        
        for i in range(0, n - m + 1):
            if get_hash_s(i, i + m) == get_hash_p(0, m):
                return i
            
            low, high = 0, m - 1
            while low < high:
                mid = (low + high) // 2
                len_pre = mid + 1
                if get_hash_s(i, i + len_pre) == get_hash_p(0, len_pre):
                    low = mid + 1
                else:
                    high = mid
            
            if get_hash_s(i + low + 1, i + m) == get_hash_p(low + 1, m):
                return i
        
        return -1