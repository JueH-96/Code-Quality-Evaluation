class Solution:
    def hasSameDigits(self, s: str) -> bool:
        # The variable 's' will store the current state of the string of digits.
        # Its value will be updated in each iteration of the while loop.

        while len(s) > 2:
            # This list will hold the characters of the next version of the string 's'.
            new_s_chars = [] 
            
            # Iterate through all pairs of consecutive digits in the current string 's'.
            # If s has length N, there are N-1 such pairs.
            # The loop iterates from i = 0 to len(s) - 2.
            # For each i, we consider s[i] and s[i+1].
            for i in range(len(s) - 1):
                # Convert characters to integers for arithmetic.
                digit1 = int(s[i])
                digit2 = int(s[i+1])
                
                # Calculate the sum modulo 10.
                sum_mod_10 = (digit1 + digit2) % 10
                
                # Convert the resulting digit back to a character and append to the list.
                new_s_chars.append(str(sum_mod_10))
            
            # Join the characters to form the new string for 's'.
            # The length of the new string will be len(s) - 1.
            s = "".join(new_s_chars) 
        
        # The loop terminates when len(s) is no longer greater than 2.
        # Given that len(s) decreases by 1 in each step and starts at >= 3 (as per constraints),
        # len(s) will be exactly 2 when the loop terminates.
        
        # Return true if the two digits in the final string 's' are the same, false otherwise.
        # Python's boolean True/False correspond to true/false in the problem statement.
        return s[0] == s[1]