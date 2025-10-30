class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # KMP helper to find all start positions of pattern pat in text s
        def kmp_search(s, pat):
            n, m = len(s), len(pat)
            if m == 0:
                return list(range(n + 1))
            # build lps
            lps = [0] * m
            length = 0
            i = 1
            while i < m:
                if pat[i] == pat[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            # search
            res = []
            i = j = 0
            while i < n:
                if s[i] == pat[j]:
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

        n = len(s)
        # split pattern p into A*B*C
        i1 = p.find('*')
        i2 = p.find('*', i1 + 1)
        A = p[:i1]
        B = p[i1+1:i2]
        C = p[i2+1:]
        lenA, lenB, lenC = len(A), len(B), len(C)

        # find all start positions of A in s (or all if empty)
        if lenA > 0:
            pref_starts = kmp_search(s, A)
        else:
            # empty prefix can start at any index 0..n
            pref_starts = list(range(n+1))

        INF = 10**18

        # build nextB[j]: minimal end index of B-match if B starts at >= j
        nextB = [INF] * (n + 2)
        if lenB > 0:
            bestB = [INF] * (n + 1)
            for st in kmp_search(s, B):
                bestB[st] = min(bestB[st], st + lenB - 1)
            # suffix-min to fill nextB
            cur = INF
            for j in range(n, -1, -1):
                if j <= n:
                    cur = min(cur, bestB[j])
                nextB[j] = cur
        else:
            # empty B: matches immediately with end = j-1
            for j in range(n+1):
                nextB[j] = j - 1

        # build nextC[j]: minimal end index of C-match if C starts at >= j
        nextC = [INF] * (n + 2)
        if lenC > 0:
            bestC = [INF] * (n + 1)
            for st in kmp_search(s, C):
                bestC[st] = min(bestC[st], st + lenC - 1)
            cur = INF
            for j in range(n, -1, -1):
                if j <= n:
                    cur = min(cur, bestC[j])
                nextC[j] = cur
        else:
            # empty C
            for j in range(n+1):
                nextC[j] = j - 1

        ans = INF
        # iterate over each possible prefix match start
        for i in pref_starts:
            # compute prefix end
            endA = i + lenA - 1
            # next we need B starting at >= endA+1
            j = endA + 1
            if j > n:
                # no room for B or C, but if B and C empty, this is ok
                endB = endA
            else:
                endB = nextB[j]
            if endB >= INF//2:
                continue
            # now C must start at >= endB+1
            k_idx = endB + 1
            if k_idx > n:
                endC = endB
            else:
                endC = nextC[k_idx]
            if endC >= INF//2:
                continue
            # total substring is s[i..endC]
            length = endC - i + 1
            if length < ans:
                ans = length

        return ans if ans < INF//2 else -1