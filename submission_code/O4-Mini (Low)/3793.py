class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        # Helper to test if a string is palindrome
        def is_palindrome(x: str) -> bool:
            i, j = 0, len(x) - 1
            while i < j:
                if x[i] != x[j]:
                    return False
                i += 1
                j -= 1
            return True

        n, m = len(s), len(t)
        # Generate all substrings of s, including the empty substring
        s_subs = ['']
        for i in range(n):
            for j in range(i, n):
                s_subs.append(s[i:j+1])
        # Generate all substrings of t, including the empty substring
        t_subs = ['']
        for i in range(m):
            for j in range(i, m):
                t_subs.append(t[i:j+1])

        best = 1  # at minimum, a single character is a palindrome
        # Try all combinations of s_sub + t_sub
        for ss in s_subs:
            for tt in t_subs:
                combo = ss + tt
                l = len(combo)
                # Only bother checking if possible to beat current best
                if l > best and is_palindrome(combo):
                    best = l
        return best