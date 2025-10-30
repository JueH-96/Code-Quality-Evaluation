class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # Find the two '*' positions
        fs = p.find('*')
        ss = p.find('*', fs + 1)
        # Split pattern into three literal parts
        A = p[:fs]
        B = p[fs+1:ss]
        C = p[ss+1:]
        
        n = len(s)
        lenA, lenB, lenC = len(A), len(B), len(C)
        
        # KMP-based search for all occurrences of a non-empty pattern
        def kmp_search(text: str, pattern: str):
            m = len(pattern)
            n2 = len(text)
            if m == 0:
                # empty pattern matches at every position 0..n
                return list(range(n2+1))
            if m > n2:
                return []
            # build prefix-function
            pi = [0]*m
            j = 0
            for i in range(1, m):
                while j > 0 and pattern[i] != pattern[j]:
                    j = pi[j-1]
                if pattern[i] == pattern[j]:
                    j += 1
                pi[i] = j
            # search
            res = []
            j = 0
            for i in range(n2):
                while j > 0 and text[i] != pattern[j]:
                    j = pi[j-1]
                if text[i] == pattern[j]:
                    j += 1
                if j == m:
                    res.append(i - m + 1)
                    j = pi[j-1]
            return res
        
        # All start positions where A, B, C occur in s
        SA = kmp_search(s, A) if lenA > 0 else list(range(n+1))
        SB = kmp_search(s, B) if lenB > 0 else list(range(n+1))
        # For C we only need the sorted list (or handle empty specially)
        if lenC > 0:
            SC = kmp_search(s, C)
        else:
            SC = []
        
        from bisect import bisect_left
        INF = n + 1
        
        # mark_B[i] = the minimal end‐index of a valid C-match
        # if B is matched at position i
        mark_B = [INF] * (n+1)
        
        if lenC > 0:
            # for each B at i, find the earliest C-start ≥ i+lenB
            for i in SB:
                k = i + lenB
                if k > n:
                    continue
                j_idx = bisect_left(SC, k)
                if j_idx < len(SC):
                    end = SC[j_idx] + lenC  # endpoint of C
                    if end < mark_B[i]:
                        mark_B[i] = end
        else:
            # C is empty: it matches immediately at k = i+lenB
            for i in SB:
                k = i + lenB
                if k <= n and k < mark_B[i]:
                    mark_B[i] = k
        
        # best_B[t] = minimum mark_B[i] over all i ≥ t
        best_B = [INF] * (n+1)
        curr = INF
        for t in range(n, -1, -1):
            if mark_B[t] < curr:
                curr = mark_B[t]
            best_B[t] = curr
        
        # Now for each A-match at l, we need a B-start ≥ l+lenA,
        # and then C-end = best_B[l+lenA]. The substring is [l, end).
        ans = INF
        for l in SA:
            t = l + lenA
            if t > n:
                continue
            end = best_B[t]
            if end < INF:
                length = end - l
                if length < ans:
                    ans = length
        
        return ans if ans < INF else -1