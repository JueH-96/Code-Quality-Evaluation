class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the string
        char_count = Counter(s)
        
        # The minimum length of t will be the maximum frequency of any character
        # because we need at least that many characters to form the anagrams.
        return max(char_count.values())