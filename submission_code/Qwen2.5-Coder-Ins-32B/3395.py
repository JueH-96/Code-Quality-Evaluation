class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the string
        char_count = Counter(s)
        
        # The minimum length of the string t is the maximum frequency of any character in s
        # This is because each anagram must contain at least one of each character in t
        return max(char_count.values())