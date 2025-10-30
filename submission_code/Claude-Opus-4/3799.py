class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        from collections import Counter
        
        # Count frequency of each digit
        digit_count = Counter(digits)
        
        count = 0
        
        # Check all 3-digit even numbers
        for num in range(100, 1000):
            if num % 2 == 0:  # Check if even
                # Extract digits of the number
                hundreds = num // 100
                tens = (num // 10) % 10
                units = num % 10
                
                # Count how many of each digit we need
                needed = Counter([hundreds, tens, units])
                
                # Check if we have enough of each digit
                can_form = True
                for digit, need_count in needed.items():
                    if digit_count[digit] < need_count:
                        can_form = False
                        break
                
                if can_form:
                    count += 1
        
        return count