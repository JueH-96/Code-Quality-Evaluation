class Solution:
    def minimumLength(self, s: str) -> int:
        from collections import Counter
        
        count = Counter(s)
        result = 0
        
        for freq in count.values():
            if freq % 2 == 1:  # odd frequency
                result += 1
            else:  # even frequency  
                result += 2
                
        return result