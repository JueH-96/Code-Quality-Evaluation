class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        """
        We want the smallest index i such that s[i .. i+len(pattern)-1]
        has at most one mismatch with pattern. Equivalently, the substring
        matches pattern in all but possibly one position.

        A classic way to solve "at most 1 mismatch" in O(n + m) time is via
        Z-functions on both the (pattern + '#' + s) and on the reversed
        strings. We then combine prefix and suffix matches to see whether
        the total matched length covers at least len(pattern)-1 characters.

        Steps:

        1. Compute Z-array for P1 = pattern + '#' + s.
           - For each i in [0..len(s)-1], prefixMatch[i] = number of matched
             consecutive characters between s[i..] and pattern's beginning.
        2. Compute Z-array for P2 = reversed(pattern) + '#' + reversed(s).
           - For each i in [0..len(s)-1], suffixMatch[i] = number of matched
             consecutive characters between s[i..] and pattern's end.
        3. For each valid starting index i in s (0 <= i <= len(s) - len(pattern)):
           let mismatch = len(pattern) - (prefixMatch[i] + suffixMatch[i+len(pattern)-1]).
           If mismatch <= 1, return i.
        4. If no valid index is found, return -1.
        """

        def z_function(st: str) -> list:
            """
            Standard Z-function calculation in O(len(st)).
            z[i] = length of the longest substring starting at st[i]
                   that matches a prefix of st.
            """
            z = [0] * len(st)
            l, r = 0, 0
            for i in range(1, len(st)):
                if i <= r:
                    z[i] = min(r - i + 1, z[i - l])
                while i + z[i] < len(st) and st[z[i]] == st[i + z[i]]:
                    z[i] += 1
                if i + z[i] - 1 > r:
                    l, r = i, i + z[i] - 1
            return z

        m, n = len(pattern), len(s)
        # If pattern is longer than s, no valid substring.
        if m > n:
            return -1

        # 1) Z-array for pattern + '#' + s
        combined1 = pattern + '#' + s
        z1 = z_function(combined1)
        prefixMatch = [0] * n
        for i in range(n):
            # index in z1 is m+1 + i
            # we cap at m because the maximum match length can't exceed pattern length
            prefixMatch[i] = min(z1[m + 1 + i], m)

        # 2) Z-array for reversed pattern + '#' + reversed s
        pattern_rev = pattern[::-1]
        s_rev = s[::-1]
        combined2 = pattern_rev + '#' + s_rev
        z2 = z_function(combined2)
        suffixMatch = [0] * n
        for i in range(n):
            # index in z2 is m+1 + (n - 1 - i)
            suffixMatch[i] = min(z2[m + 1 + (n - 1 - i)], m)

        # 3) Check for at most 1 mismatch
        for i in range(n - m + 1):
            # total matched = prefixMatch[i] + suffixMatch[i + m - 1]
            # mismatch = m - (total matched)
            if m - (prefixMatch[i] + suffixMatch[i + m - 1]) <= 1:
                return i

        return -1