class Solution:
    def minimumLength(self, s: str) -> int:
        from collections import Counter
        
        # Count frequency of each character
        freq = Counter(s)
        
        result = 0
        for count in freq.values():
            if count < 3:
                # Can't perform any operations
                result += count
            elif count % 2 == 1:
                # Odd count >= 3: can reduce to 1
                result += 1
            else:
                # Even count >= 3: can reduce to 2
                result += 2
                
        return result