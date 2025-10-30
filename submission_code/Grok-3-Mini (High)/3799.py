from typing import List

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        # Create frequency array for digits 0 to 9
        freq = [0] * 10
        for digit in digits:
            freq[digit] += 1
        
        count = 0
        
        # Iterate over all possible hundreds digits (1-9, no leading zero)
        for H in range(1, 10):
            # Iterate over all possible tens digits (0-9)
            for T in range(10):
                # Iterate over all possible units digits that are even (0,2,4,6,8)
                for U in range(0, 10, 2):
                    # Create a set of the digits chosen to handle unique digits
                    digits_set = {H, T, U}
                    valid = True
                    
                    # Check the frequency for each unique digit in the set
                    for dig in digits_set:
                        count_d = 0
                        if dig == H:
                            count_d += 1
                        if dig == T:
                            count_d += 1
                        if dig == U:
                            count_d += 1
                        if count_d > freq[dig]:
                            valid = False
                            break  # No need to check further if invalid
                    
                    # If all digits are valid based on frequency, increment count
                    if valid:
                        count += 1
        
        return count