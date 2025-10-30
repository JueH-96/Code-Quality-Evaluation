class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        max_len = 0
        n, m = len(s), len(t)
        
        # Check all substrings of s (A is non-empty, B is empty)
        for i in range(n):
            for j in range(i, n):
                a = s[i:j+1]
                if a == a[::-1]:
                    max_len = max(max_len, j - i + 1)
        
        # Check all substrings of t (A is empty, B is non-empty)
        for i in range(m):
            for j in range(i, m):
                b = t[i:j+1]
                if b == b[::-1]:
                    max_len = max(max_len, j - i + 1)
        
        # Check combinations of non-empty A and B
        for i in range(n):
            for j in range(i, n):
                a = s[i:j+1]
                a_len = j - i + 1
                a_first = a[0]
                for x in range(m):
                    for y in range(x, m):
                        b = t[x:y+1]
                        b_last = b[-1]
                        b_len = y - x + 1
                        combined_len = a_len + b_len
                        if combined_len <= max_len:
                            continue
                        if a_first != b_last:
                            continue
                        combined = a + b
                        if combined == combined[::-1]:
                            max_len = combined_len
        
        return max_len