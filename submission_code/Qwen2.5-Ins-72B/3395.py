class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the string
        char_count = Counter(s)
        
        # The minimum length of t is the maximum frequency of any character
        # because t must be able to form all anagrams in s
        return max(char_count.values())