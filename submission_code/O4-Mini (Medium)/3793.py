class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        # Generate all substrings of s (including empty) and t (including empty)
        n, m = len(s), len(t)
        s_subs = [s[i:j] for i in range(n + 1) for j in range(i, n + 1)]
        t_subs = [t[i:j] for i in range(m + 1) for j in range(i, m + 1)]
        
        max_len = 0
        # Try all pairs of substrings
        for ss in s_subs:
            for tt in t_subs:
                total_len = len(ss) + len(tt)
                # Only check if this pair can beat current max
                if total_len > max_len:
                    combo = ss + tt
                    # Check palindrome
                    if combo == combo[::-1]:
                        max_len = total_len
        return max_len