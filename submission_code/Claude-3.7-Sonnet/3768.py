class Solution:
    def hasSameDigits(self, s: str) -> bool:
        # Continue the operation until s has exactly 2 digits
        while len(s) > 2:
            # Create a new string for the next iteration
            new_s = []
            # For each consecutive pair of digits
            for i in range(len(s) - 1):
                # Calculate the sum modulo 10
                digit_sum = (int(s[i]) + int(s[i+1])) % 10
                new_s.append(str(digit_sum))
            # Update s with the new sequence of digits
            s = "".join(new_s)
        
        # Check if the final two digits are the same
        return s[0] == s[1]