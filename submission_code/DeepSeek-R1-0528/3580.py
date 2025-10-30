class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        base1, base2 = 131, 137
        mod1, mod2 = 10**9+7, 10**9+9
        n = len(s)
        m = len(pattern)
        max_len = max(n, m)
        
        pow1 = [1] * (max_len + 1)
        pow2 = [1] * (max_len + 1)
        for i in range(1, max_len + 1):
            pow1[i] = (pow1[i-1] * base1) % mod1
            pow2[i] = (pow2[i-1] * base2) % mod2
        
        h1_s = [0] * (n + 1)
        h2_s = [0] * (n + 1)
        for i in range(1, n + 1):
            val = ord(s[i-1]) - ord('a') + 1
            h1_s[i] = (h1_s[i-1] * base1 + val) % mod1
            h2_s[i] = (h2_s[i-1] * base2 + val) % mod2
        
        h1_pat = [0] * (m + 1)
        h2_pat = [0] * (m + 1)
        for i in range(1, m + 1):
            val = ord(pattern[i-1]) - ord('a') + 1
            h1_pat[i] = (h1_pat[i-1] * base1 + val) % mod1
            h2_pat[i] = (h2_pat[i-1] * base2 + val) % mod2
        
        for i in range(0, n - m + 1):
            low_len, high_len = 0, m + 1
            while low_len < high_len:
                mid_len = (low_len + high_len) // 2
                if mid_len == 0:
                    hash_pat1, hash_pat2 = 0, 0
                    hash_s1, hash_s2 = 0, 0
                else:
                    hash_pat1 = h1_pat[mid_len]
                    hash_pat2 = h2_pat[mid_len]
                    hash_s1 = (h1_s[i+mid_len] - h1_s[i] * pow1[mid_len]) % mod1
                    hash_s2 = (h2_s[i+mid_len] - h2_s[i] * pow2[mid_len]) % mod2
                if (hash_pat1 == hash_s1) and (hash_pat2 == hash_s2):
                    low_len = mid_len + 1
                else:
                    high_len = mid_len
            common_prefix_len = low_len - 1
            if common_prefix_len == m:
                return i
            suffix_length = m - common_prefix_len - 1
            if suffix_length == 0:
                return i
            start_pat = common_prefix_len + 1
            start_s = i + common_prefix_len + 1
            end_pat = start_pat + suffix_length
            end_s = start_s + suffix_length
            pat1 = (h1_pat[end_pat] - h1_pat[start_pat] * pow1[suffix_length]) % mod1
            pat2 = (h2_pat[end_pat] - h2_pat[start_pat] * pow2[suffix_length]) % mod2
            s1 = (h1_s[end_s] - h1_s[start_s] * pow1[suffix_length]) % mod1
            s2 = (h2_s[end_s] - h2_s[start_s] * pow2[suffix_length]) % mod2
            if (pat1, pat2) == (s1, s2):
                return i
        return -1