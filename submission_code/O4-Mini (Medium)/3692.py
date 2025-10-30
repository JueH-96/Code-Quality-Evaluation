import bisect

class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # KMP prefix-function
        def compute_pi(pat):
            m = len(pat)
            pi = [0] * m
            k = 0
            for i in range(1, m):
                while k > 0 and pat[k] != pat[i]:
                    k = pi[k-1]
                if pat[k] == pat[i]:
                    k += 1
                pi[i] = k
            return pi

        # KMP search: return list of start indices where pattern occurs in text
        def kmp_search(text, pat):
            n, m = len(text), len(pat)
            if m == 0:
                # empty pattern matches at every position 0..n
                return list(range(n+1))
            pi = compute_pi(pat)
            res = []
            q = 0
            for i in range(n):
                while q > 0 and pat[q] != text[i]:
                    q = pi[q-1]
                if pat[q] == text[i]:
                    q += 1
                if q == m:
                    res.append(i - m + 1)
                    q = pi[q-1]
            return res

        n = len(s)
        # Split pattern into the three literal parts around the two '*'
        parts = p.split('*')
        A, B, C = parts[0], parts[1], parts[2]
        # Find all occurrences
        occA = kmp_search(s, A)
        occB = kmp_search(s, B)
        occC = kmp_search(s, C)

        # If any non-empty segment never occurs, no match possible
        if A and not occA: return -1
        if B and not occB: return -1
        if C and not occC: return -1

        ans = float('inf')
        lenA, lenB, lenC = len(A), len(B), len(C)

        # For each possible placement of A, find the earliest B after it,
        # then the earliest C after that, to minimize total length.
        for a_start in occA:
            a_end = a_start + lenA
            # find B start >= a_end
            idxb = bisect.bisect_left(occB, a_end)
            if idxb == len(occB):
                continue
            b_start = occB[idxb]
            b_end = b_start + lenB
            # find C start >= b_end
            idxc = bisect.bisect_left(occC, b_end)
            if idxc == len(occC):
                continue
            c_start = occC[idxc]
            c_end = c_start + lenC
            # total substring from a_start to c_end-1 inclusive
            length = c_end - a_start
            if length < ans:
                ans = length

        return ans if ans != float('inf') else -1