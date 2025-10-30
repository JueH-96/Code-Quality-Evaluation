class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        def get_substrings(s):
            subs = []
            n = len(s)
            for i in range(n + 1):
                for j in range(i, n + 1):
                    subs.append(s[i:j])
            return subs
        
        s_substrings = get_substrings(s)
        t_substrings = get_substrings(t)
        
        max_length = 0
        
        for a in s_substrings:
            for b in t_substrings:
                candidate = a + b
                if candidate == candidate[::-1]:
                    current_length = len(candidate)
                    if current_length > max_length:
                        max_length = current_length
        
        return max_length