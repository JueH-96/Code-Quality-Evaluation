class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        # Collect every (possibly empty) substring of s
        subs_s = ['']                       # the empty substring
        m = len(s)
        for i in range(m):
            for j in range(i + 1, m + 1):   # j is exclusive
                subs_s.append(s[i:j])

        # Collect every (possibly empty) substring of t
        subs_t = ['']
        n = len(t)
        for i in range(n):
            for j in range(i + 1, n + 1):
                subs_t.append(t[i:j])

        # Helper to test palindromicity
        def is_pal(st: str) -> bool:
            return st == st[::-1]

        best = 0
        # Try every pair (sub from s, sub from t)
        for a in subs_s:
            len_a = len(a)
            for b in subs_t:
                total_len = len_a + len(b)
                # no need to check if we can't beat current best
                if total_len <= best:
                    continue
                if is_pal(a + b):
                    best = total_len
        return best