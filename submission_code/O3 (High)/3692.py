class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # ---------- helpers ----------
        def build_lps(pattern: str):
            """KMP prefix table"""
            m = len(pattern)
            lps = [0] * m
            length = 0
            i = 1
            while i < m:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            return lps

        def kmp_occurrences(text: str, pattern: str):
            """returns list of starting indices where pattern occurs in text"""
            m = len(pattern)
            if m == 0 or m > len(text):
                return []
            lps = build_lps(pattern)
            res = []
            i = j = 0
            n = len(text)
            while i < n:
                if text[i] == pattern[j]:
                    i += 1
                    j += 1
                    if j == m:
                        res.append(i - m)
                        j = lps[j - 1]
                else:
                    if j:
                        j = lps[j - 1]
                    else:
                        i += 1
            return res
        # ---------- parse pattern ----------
        parts = p.split('*')
        # exactly two '*', hence three parts (possibly empty)
        A, B, C = parts[0], parts[1], parts[2]
        lenA, lenB, lenC = len(A), len(B), len(C)
        n = len(s)
        INF = 10 ** 18

        # ---------- pre–compute nextC array ----------
        if lenC == 0:
            # empty C occurs at every index, including position n (empty suffix)
            nextC = list(range(n + 1))
        else:
            isCStart = [False] * n
            for idx in kmp_occurrences(s, C):
                isCStart[idx] = True
            nextC = [INF] * (n + 1)
            earliest = INF
            for i in range(n - 1, -1, -1):
                if isCStart[i]:
                    earliest = i
                nextC[i] = earliest
            nextC[n] = INF  # out-of-range sentinel

        # ---------- pre–compute isBStart ----------
        if lenB == 0:
            # empty B starts at every index (and at n, but we never access it)
            isBStart = None  # flag to signal special handling
        else:
            isBStart = [False] * n
            for idx in kmp_occurrences(s, B):
                isBStart[idx] = True

        # ---------- compute bcEnd[i]: earliest end (exclusive) of a substring
        # starting at i that matches B* C ----------
        bcEnd = [INF] * (n + 1)
        bcEnd[n] = INF
        for i in range(n - 1, -1, -1):
            candidate_end = INF
            if lenB == 0:
                # B empty – it starts right at i
                posC = nextC[i]
                if posC != INF:
                    candidate_end = posC + lenC
            else:
                if isBStart[i]:
                    after_B = i + lenB
                    if after_B <= n:
                        posC = nextC[after_B]
                        if posC != INF:
                            candidate_end = posC + lenC
            # we can skip current character via the star preceding B
            bcEnd[i] = min(candidate_end, bcEnd[i + 1])

        # ---------- evaluate answer ----------
        ans = INF
        if lenA == 0:
            # substring may start anywhere; minimal length for each i is bcEnd[i]-i
            for i in range(n):
                if bcEnd[i] != INF:
                    ans = min(ans, bcEnd[i] - i)
        else:
            # enumerate occurrences of A
            for posA in kmp_occurrences(s, A):
                start_after_A = posA + lenA
                if start_after_A <= n and bcEnd[start_after_A] != INF:
                    ans = min(ans, bcEnd[start_after_A] - posA)

        return -1 if ans == INF else ans