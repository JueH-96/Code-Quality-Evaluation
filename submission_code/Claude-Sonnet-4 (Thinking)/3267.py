class Solution:
    def maximumLength(self, s: str) -> int:
        from collections import defaultdict
        
        # Count occurrences of all special substrings
        count = defaultdict(int)
        n = len(s)
        
        for i in range(n):
            for j in range(i, n):
                substring = s[i:j+1]
                # Check if substring is special (all characters are the same)
                if len(set(substring)) == 1:
                    count[substring] += 1
        
        # Find the longest special substring that occurs at least 3 times
        max_length = -1
        for substring, freq in count.items():
            if freq >= 3:
                max_length = max(max_length, len(substring))
        
        return max_length