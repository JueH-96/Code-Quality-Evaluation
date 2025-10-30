class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        n = len(digits)
        result = set()
        
        # Try all possible combinations of three digits
        for i in range(n):
            # Skip if first digit is 0 (no leading zeros allowed)
            if digits[i] == 0:
                continue
                
            for j in range(n):
                # Skip if using same position
                if j == i:
                    continue
                    
                for k in range(n):
                    # Skip if using same position
                    if k == i or k == j:
                        continue
                    
                    # Check if last digit is even
                    if digits[k] % 2 == 0:
                        # Form the number
                        num = digits[i] * 100 + digits[j] * 10 + digits[k]
                        result.add(num)
        
        return len(result)