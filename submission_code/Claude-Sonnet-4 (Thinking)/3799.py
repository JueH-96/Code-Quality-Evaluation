class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        from collections import Counter
        
        freq = Counter(digits)
        count = 0
        
        # Try each even digit as the last digit
        for last in [0, 2, 4, 6, 8]:
            if freq[last] == 0:
                continue
                
            freq[last] -= 1  # Use this digit for last position
            
            # Try each non-zero digit as the first digit
            for first in range(1, 10):
                if freq[first] == 0:
                    continue
                    
                freq[first] -= 1  # Use this digit for first position
                
                # For middle position, count all remaining digits
                count += sum(freq.values())
                
                freq[first] += 1  # Restore first digit
                
            freq[last] += 1  # Restore last digit
        
        return count