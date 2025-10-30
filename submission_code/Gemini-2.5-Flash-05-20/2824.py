class Solution:
    def isFascinating(self, n: int) -> bool:
        # Step 1: Calculate 2*n and 3*n
        n2 = 2 * n
        n3 = 3 * n

        # Step 2: Concatenate n, 2*n, and 3*n into a string
        s = str(n) + str(n2) + str(n3)

        # Step 3: Validate Digits
        # A fascinating number must result in a 9-digit string (1-9, exactly once)
        if len(s) != 9:
            return False

        # Use a set to keep track of seen digits to check for uniqueness
        seen_digits = set()

        for char_digit in s:
            # Condition: Does not contain any 0's
            if char_digit == '0':
                return False
            
            digit = int(char_digit)
            
            # Condition: Contains all digits from 1 to 9 exactly once
            if digit in seen_digits:
                return False # Digit already seen, not unique
            
            seen_digits.add(digit)
        
        # If we reach here, it means:
        # 1. The string length was 9.
        # 2. No '0's were found.
        # 3. All 9 characters were unique and non-zero.
        # This implies that all digits from 1 to 9 must have been present exactly once.
        return True