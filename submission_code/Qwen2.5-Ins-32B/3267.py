class Solution:
    def maximumLength(self, s: str) -> int:
        from collections import defaultdict
        
        def get_substrings(s):
            substrings = defaultdict(int)
            n = len(s)
            i = 0
            while i < n:
                j = i
                while j < n and s[j] == s[i]:
                    j += 1
                length = j - i
                for k in range(1, length + 1):
                    substrings[s[i] * k] += length - k + 1
                i = j
            return substrings
        
        substrings = get_substrings(s)
        max_length = -1
        for substring, count in substrings.items():
            if count >= 3:
                max_length = max(max_length, len(substring))
        
        return max_length