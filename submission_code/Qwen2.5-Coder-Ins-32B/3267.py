class Solution:
    def maximumLength(self, s: str) -> int:
        from collections import defaultdict
        
        def generate_special_substrings(s):
            n = len(s)
            special_substrings = defaultdict(int)
            for i in range(n):
                for j in range(i + 1, n + 1):
                    substring = s[i:j]
                    if len(set(substring)) == 1:
                        special_substrings[substring] += 1
            return special_substrings
        
        special_substrings = generate_special_substrings(s)
        max_length = -1
        
        for substring, count in special_substrings.items():
            if count >= 3:
                max_length = max(max_length, len(substring))
        
        return max_length