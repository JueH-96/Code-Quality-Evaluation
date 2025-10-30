class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        from collections import Counter
        
        # Count frequency of each digit
        count = Counter(digits)
        
        result = 0
        
        # Try all possible combinations for hundreds, tens, and units place
        for units in range(0, 10, 2):  # Even digits only for units place
            if count[units] == 0:
                continue
                
            # Use one copy of units digit
            count[units] -= 1
            
            for hundreds in range(1, 10):  # No leading zero
                if count[hundreds] == 0:
                    continue
                    
                # Use one copy of hundreds digit
                count[hundreds] -= 1
                
                for tens in range(0, 10):
                    if count[tens] == 0:
                        continue
                        
                    # We have a valid combination
                    result += 1
                
                # Restore hundreds digit
                count[hundreds] += 1
            
            # Restore units digit
            count[units] += 1
        
        return result