class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        # Helper function to generate all possible substrings including empty
        def generate_substrings(string):
            n = len(string)
            substrings = []
            for i in range(n + 1):
                for j in range(i, n + 1):
                    substrings.append(string[i:j])
            return substrings
        
        # Generate all substrings for s and t
        s_substrings = generate_substrings(s)
        t_substrings = generate_substrings(t)
        
        max_length = 0
        
        # Check all possible pairs of substrings
        for sub_s in s_substrings:
            for sub_t in t_substrings:
                candidate = sub_s + sub_t
                if candidate == candidate[::-1]:
                    max_length = max(max_length, len(candidate))
        
        return max_length