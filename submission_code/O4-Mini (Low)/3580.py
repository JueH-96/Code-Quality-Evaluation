class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        # If pattern length is 1, any single character substring can be changed in at most one place
        # so the answer is always 0 (smallest index).
        m, n = len(pattern), len(s)
        if m == 1:
            return 0
        
        # Z-algorithm to compute Z-array for string str_
        def compute_z(str_):
            L = len(str_)
            Z = [0] * L
            l, r = 0, 0
            for i in range(1, L):
                if i <= r:
                    # mirror trick
                    k = i - l
                    Z[i] = min(r - i + 1, Z[k])
                # attempt to extend
                while i + Z[i] < L and str_[Z[i]] == str_[i + Z[i]]:
                    Z[i] += 1
                # update window
                if i + Z[i] - 1 > r:
                    l, r = i, i + Z[i] - 1
            return Z
        
        # Build concatenation for prefix matches: pattern + '$' + s
        sep = '$'
        cat1 = pattern + sep + s
        Z1 = compute_z(cat1)
        
        # Build concatenation for suffix matches: reverse(pattern) + '$' + reverse(s)
        rp = pattern[::-1]
        rs = s[::-1]
        cat2 = rp + sep + rs
        Z2 = compute_z(cat2)
        
        # We will slide window start i from 0..n-m
        # For each i, prefix-match length is Z1[m+1 + i].
        # If that == m, zero mismatches => return i.
        # Otherwise let l = Z1[...], that's first mismatch position.
        # We need the suffix match length for the tail after l:
        #   need Z2[m+1 + (n - i - m)] >= m - l - 1
        for i in range(n - m + 1):
            zpref = Z1[m + 1 + i]
            if zpref >= m:
                # perfect match
                return i
            # one mismatch at position l = zpref
            # need to check that the rest matches: suffix length >= m - l - 1
            needed = m - zpref - 1
            # index in Z2 for suffix at window i:
            # reversed substring in rs starts at pos (n-1 - (i+m-1)) = n - i - m
            idx2 = m + 1 + (n - i - m)
            if idx2 < len(Z2) and Z2[idx2] >= needed:
                return i
        
        return -1