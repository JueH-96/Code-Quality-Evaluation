class Solution:
    def maximumLength(self, s: str) -> int:
        # Dictionary to count occurrences of each special substring
        special_count = {}
        n = len(s)
        
        # Generate all possible substrings
        for i in range(n):
            for j in range(i + 1, n + 1):
                substring = s[i:j]
                
                # Check if substring is special (all characters are the same)
                if len(set(substring)) == 1:
                    special_count[substring] = special_count.get(substring, 0) + 1
        
        # Find the maximum length of special substrings that occur at least 3 times
        max_length = -1
        for substring, count in special_count.items():
            if count >= 3:
                max_length = max(max_length, len(substring))
        
        return max_length