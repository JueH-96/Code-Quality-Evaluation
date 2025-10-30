class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        m = len(pattern)
        n = len(s)
        if m >= n:
            return -1
        
        # Compute prefix matches using Z-array
        concat_pre = pattern + '#' + s
        z_pre = self.compute_z(concat_pre)
        pre = z_pre[len(pattern) + 1:]  # pre[i] is prefix match length starting at s[i]
        
        # Compute suffix matches using reversed strings
        rev_pattern = pattern[::-1]
        rev_s = s[::-1]
        concat_suf = rev_pattern + '#' + rev_s
        z_suf = self.compute_z(concat_suf)
        suf_rev = z_suf[len(rev_pattern) + 1:]  # z_suf for each position in rev_s
        
        # Build suf array for original s
        suf = [0] * n  # suf[i] is the max suffix of pattern matching ending at i
        for j_in_rev_s in range(len(suf_rev)):
            original_i = n - 1 - j_in_rev_s
            suf_val = suf_rev[j_in_rev_s]
            suf[original_i] = min(suf_val, m)
        
        # Check each possible starting index
        for i in range(n - m + 1):
            # Case 1: entire pattern matches
            if pre[i] >= m:
                return i
            # Case 2: check prefix + suffix
            end_pos = i + m - 1
            if end_pos >= n:
                continue
            current_suf = suf[end_pos]
            if (pre[i] + current_suf) >= (m - 1):
                return i
        
        return -1
    
    def compute_z(self, s: str) -> list:
        if not s:
            return []
        n = len(s)
        Z = [0] * n
        l, r = 0, 0
        
        for i in range(1, n):
            # Case where i is outside the current Z box
            if i > r:
                l = r = i
                while r < n and s[r - l] == s[r]:
                    r += 1
                Z[i] = r - l
                r -= 1
            else:
                k = i - l
                # Case where we can copy value from Z[k]
                if Z[k] < r - i + 1:
                    Z[i] = Z[k]
                else:
                    # Need to extend the Z box
                    l = i
                    while r < n and s[r - l] == s[r]:
                        r += 1
                    Z[i] = r - l
                    r -= 1
        
        return Z