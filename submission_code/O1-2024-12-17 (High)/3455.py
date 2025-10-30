class Solution:
    def minimumLength(self, s: str) -> int:
        # Count frequency of each character
        freq = [0]*26
        for ch in s:
            freq[ord(ch) - ord('a')] += 1
        
        # distinct = how many characters appear at least once
        # even_count = how many of those characters appear an even number of times
        distinct = 0
        even_count = 0
        for f in freq:
            if f > 0:
                distinct += 1
                if f % 2 == 0:
                    even_count += 1
        
        # Minimum final length = number of distinct chars + number of even-frequency chars
        return distinct + even_count