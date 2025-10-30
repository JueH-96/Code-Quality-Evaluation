class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        from collections import Counter
        
        # Count frequency of each digit
        freq = Counter(digits)
        
        count = 0
        
        # Check all three-digit even numbers
        for num in range(100, 1000, 2):  # Only even numbers
            # Extract digits
            a = num // 100
            b = (num // 10) % 10
            c = num % 10
            
            # Count required digits for this number
            required = Counter([a, b, c])
            
            # Check if we have enough of each digit
            can_form = True
            for digit, need in required.items():
                if freq.get(digit, 0) < need:
                    can_form = False
                    break
            
            if can_form:
                count += 1
        
        return count