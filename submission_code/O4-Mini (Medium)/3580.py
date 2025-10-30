class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        # Z-algorithm
        def compute_z(s: str):
            n = len(s)
            z = [0] * n
            l = r = 0
            for i in range(1, n):
                if i <= r:
                    z[i] = min(r - i + 1, z[i - l])
                while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                    z[i] += 1
                if i + z[i] - 1 > r:
                    l, r = i, i + z[i] - 1
            z[0] = n
            return z

        n, m = len(s), len(pattern)
        # Edge case: if pattern length is 1, any single char substring can match with ≤1 change
        if m == 1:
            return 0

        # Build Z-array for pattern + '#' + s
        sep = '#'
        combined1 = pattern + sep + s
        z1 = compute_z(combined1)
        # Build Z-array for reversed pattern + '#' + reversed s
        s_rev = s[::-1]
        p_rev = pattern[::-1]
        combined2 = p_rev + sep + s_rev
        z2 = compute_z(combined2)

        # For each window start i in s[0..n-m], check mismatches <=1
        # pre = length of prefix match at i
        # suf = length of suffix match after possible one mismatch
        for i in range(0, n - m + 1):
            pre = z1[m + 1 + i]
            if pre >= m:
                # perfect match, zero changes needed
                return i
            # suffix match: index in reversed string corresponding to window start i
            i_rev = n - i - m
            suf = z2[m + 1 + i_rev]
            # if pre + suf >= m-1, we can cover all but one character by matches
            if pre + suf >= m - 1:
                return i

        return -1