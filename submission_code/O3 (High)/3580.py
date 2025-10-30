class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        """
        Return the smallest index i such that s[i : i+len(pattern)] can be turned
        into pattern by changing at most one character.  If no such index exists
        return -1.
        """
        n, m = len(s), len(pattern)
        if m > n:
            return -1                            # should not happen according to constraints

        # ----------  Z-function  ----------
        def z_function(st: str) -> list:
            """
            z[i] = length of the longest substring starting at i
                    which is also a prefix of st
            """
            z = [0] * len(st)
            l = r = 0
            for i in range(1, len(st)):
                if i <= r:
                    z[i] = min(r - i + 1, z[i - l])
                while i + z[i] < len(st) and st[z[i]] == st[i + z[i]]:
                    z[i] += 1
                if i + z[i] - 1 > r:
                    l, r = i, i + z[i] - 1
            return z

        # ----------  longest common PREFIX between pattern and every suffix of s ----------
        comb_pref = pattern + '#' + s
        z_pref = z_function(comb_pref)
        offset_pref = m + 1                       # starting position of s in comb_pref
        pref_match = [0] * (n - m + 1)            # pref_match[i] = lcp(pattern, s[i:])
        for i in range(n - m + 1):
            pref_match[i] = min(z_pref[offset_pref + i], m)

        # ----------  longest common SUFFIX between pattern and every substring of s ----------
        #   Use the same trick on the reversed strings:  LCSuffix == LCPrefix on reversed strings
        rs = s[::-1]
        rp = pattern[::-1]
        comb_suf = rp + '#' + rs
        z_suf = z_function(comb_suf)
        offset_suf = m + 1
        suf_match = [0] * (n - m + 1)             # suf_match[i] = lcs(pattern, s[i : i+m])
        for i in range(n - m + 1):
            # position in the reversed string where the reversed window starts
            rev_idx = n - (i + m)
            suf_match[i] = min(z_suf[offset_suf + rev_idx], m)

        # ----------  decide ----------
        for i in range(n - m + 1):
            # 0 mismatches
            if pref_match[i] == m:
                return i
            # at most 1 mismatch  <=>  prefix + suffix ≥ m-1
            if pref_match[i] + suf_match[i] >= m - 1:
                return i
        return -1