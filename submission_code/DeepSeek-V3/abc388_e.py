class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        max_len = 0
        # Check all possible s1 + t1 combinations where s1 is a substring of s and t1 is a substring of t
        # The combined string s1 + t1 should be a palindrome
        # Generate all possible substrings of s and t, then check their concatenations
        # Since the strings are up to 30 characters, this might be O(n^4) in the worst case, but manageable
        
        # Precompute all substrings of s and t along with their start and end indices (or just the strings)
        # Alternatively, iterate through all possible substrings in s and t
        
        len_s = len(s)
        len_t = len(t)
        
        # Check all possible s_sub and t_sub combinations
        for i in range(len_s + 1):
            for j in range(i, len_s + 1):
                s_sub = s[i:j]
                for k in range(len_t + 1):
                    for l in range(k, len_t + 1):
                        t_sub = t[k:l]
                        candidate = s_sub + t_sub
                        if candidate == candidate[::-1]:
                            current_len = len(candidate)
                            if current_len > max_len:
                                max_len = current_len
        return max_len if max_len != 0 else 0