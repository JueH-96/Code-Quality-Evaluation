class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        m = n // 2
        bad_pairs = [0] * m
        for i in range(m):
            if s[i] != s[2 * m - 1 - i]:
                bad_pairs[i] = 1
        prefix_bad = [0] * (m + 1)
        for i in range(m):
            prefix_bad[i + 1] = prefix_bad[i] + bad_pairs[i]
        res = []
        for a, b, c, d in queries:
            lower1 = 0
            upper1 = a
            lower2 = b + 1
            upper2 = m
            threshold1 = 2 * m - 1 - c
            threshold2 = 2 * m - d - 2
            # Case 1: i in [0, a) and (i > threshold1 or i <= threshold2)
            l = 0
            r = upper1
            if threshold1 < m:
                l = max(l, threshold1 + 1)
            l = max(l, 0)
            r = min(r, m)
            count1 = prefix_bad[r] - prefix_bad[l]
            l = 0
            r = upper1
            if threshold2 >= 0:
                r = min(r, threshold2 + 1)
            l = max(l, 0)
            r = min(r, m)
            count1 += prefix_bad[r] - prefix_bad[l]
            # Case 2: i in [b+1, m-1] and (i > threshold1 or i <= threshold2)
            l = lower2
            r = upper2
            if threshold1 < m:
                l = max(l, threshold1 + 1)
            l = max(l, b + 1)
            r = min(r, m)
            count2 = prefix_bad[r] - prefix_bad[l]
            l = lower2
            r = upper2
            if threshold2 >= 0:
                r = min(r, threshold2 + 1)
            l = max(l, b + 1)
            r = min(r, m)
            count2 += prefix_bad[r] - prefix_bad[l]
            total = count1 + count2
            if total == 0:
                res.append(True)
            else:
                res.append(False)
        return res