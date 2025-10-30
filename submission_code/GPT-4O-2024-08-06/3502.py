class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        def is_valid(substring, k):
            # Check if any character in the substring appears at least k times
            from collections import Counter
            count = Counter(substring)
            for char in count:
                if count[char] >= k:
                    return True
            return False
        
        n = len(s)
        total_substrings = 0
        
        # Generate all possible substrings
        for start in range(n):
            for end in range(start + 1, n + 1):
                substring = s[start:end]
                if is_valid(substring, k):
                    total_substrings += 1
        
        return total_substrings