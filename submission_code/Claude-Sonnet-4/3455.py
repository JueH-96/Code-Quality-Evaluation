class Solution:
    def minimumLength(self, s: str) -> int:
        from collections import Counter
        
        # Count frequency of each character
        char_count = Counter(s)
        
        result = 0
        for count in char_count.values():
            if count % 2 == 1:
                result += 1  # odd count reduces to 1
            else:
                result += 2  # even count reduces to 2
                
        return result