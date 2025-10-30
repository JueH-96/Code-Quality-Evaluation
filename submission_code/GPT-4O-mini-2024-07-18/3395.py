class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        
        # Count the frequency of each character in the string
        char_count = Counter(s)
        
        # The minimum length of t is the number of unique characters
        # divided by the maximum frequency of any character
        max_freq = max(char_count.values())
        
        return sum(char_count.values()) // max_freq