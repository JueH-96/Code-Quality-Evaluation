class Solution:
    def maximumLength(self, s: str) -> int:
        from collections import defaultdict
        
        max_length = -1
        n = len(s)
        
        # Check all possible lengths of special substrings
        for length in range(1, n + 1):
            count = defaultdict(int)
            
            # Count occurrences of each special substring of the current length
            for i in range(n - length + 1):
                substring = s[i:i + length]
                if len(set(substring)) == 1:  # Check if the substring is special
                    count[substring] += 1
            
            # Check if any special substring of the current length occurs at least thrice
            for key in count:
                if count[key] >= 3:
                    max_length = length
        
        return max_length