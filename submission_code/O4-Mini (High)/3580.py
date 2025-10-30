class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        # Z-algorithm to compute longest common prefix lengths
        def z_algo(t: str):
            n = len(t)
            z = [0] * n
            l = 0
            r = 0
            for i in range(1, n):
                if i <= r:
                    # reuse previously computed value
                    z[i] = min(r - i + 1, z[i - l])
                # extend match from position i
                while i + z[i] < n and t[z[i]] == t[i + z[i]]:
                    z[i] += 1
                # update [l, r] if we extended past r
                if i + z[i] - 1 > r:
                    l = i
                    r = i + z[i] - 1
            return z

        n, m = len(s), len(pattern)
        # Build Z-array for pattern + '#' + s to get prefix matches
        t1 = pattern + '#' + s
        z1 = z_algo(t1)
        # Build Z-array for reversed pattern + '#' + reversed s to get suffix matches
        rev_pat = pattern[::-1]
        rev_s = s[::-1]
        t2 = rev_pat + '#' + rev_s
        z2 = z_algo(t2)

        # Slide window of length m over s
        for i in range(n - m + 1):
            # Number of matching chars at the start
            lcp = z1[m + 1 + i]
            # If full match, zero mismatches => valid
            if lcp == m:
                return i
            # Compute how many chars at the end match after one allowed mismatch
            rev_i = n - i - m
            suf = z2[m + 1 + rev_i]
            # After the single mismatch at position i+lcp, we need the rest (m-lcp-1) to match
            if suf >= m - lcp - 1:
                return i

        # No valid starting index found
        return -1