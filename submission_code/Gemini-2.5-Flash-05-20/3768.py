class Solution:
    def hasSameDigits(self, s: str) -> bool:
        # Continue performing operations until the string s has exactly two digits.
        while len(s) > 2:
            next_s_list = []  # List to store digits of the new string
            
            # Iterate through all consecutive pairs of digits in the current string s.
            # The loop runs up to len(s) - 1 because we need s[i] and s[i+1].
            for i in range(len(s) - 1):
                # Convert the current two character digits to integers.
                digit1 = int(s[i])
                digit2 = int(s[i+1])
                
                # Calculate the sum modulo 10 to get the new digit.
                new_digit = (digit1 + digit2) % 10
                
                # Convert the new digit back to a string and append it to our list.
                next_s_list.append(str(new_digit))
            
            # Join the list of new digits to form the updated string s.
            s = "".join(next_s_list)
        
        # After the loop terminates, s will contain exactly two digits.
        # Check if these two digits are the same.
        return s[0] == s[1]