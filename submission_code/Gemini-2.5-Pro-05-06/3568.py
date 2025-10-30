class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        
        # Step 1: Convert numbers to strings and pad with leading zeros to length 4.
        # The str.zfill(width) method pads a string on the left with zeros 
        # to reach the specified width.
        s_num1 = str(num1).zfill(4)
        s_num2 = str(num2).zfill(4)
        s_num3 = str(num3).zfill(4)

        key_string_builder = [] # Used to collect the digits of the key as strings

        # Step 2: Generate the key's digits.
        # The key is a four-digit number. We iterate four times, once for each digit position.
        # String indexing is 0-based, so positions 0, 1, 2, 3 correspond to the 
        # 1st, 2nd, 3rd, and 4th digits respectively.
        for i in range(4):
            # Extract the character at the current position i from each padded string.
            # These characters represent digits.
            char_digit_num1 = s_num1[i]
            char_digit_num2 = s_num2[i]
            char_digit_num3 = s_num3[i]
            
            # Convert these character digits to integers to perform numerical comparison.
            digit_from_num1 = int(char_digit_num1)
            digit_from_num2 = int(char_digit_num2)
            digit_from_num3 = int(char_digit_num3)
            
            # Find the smallest digit among the three for the current position.
            smallest_digit_at_pos_i = min(digit_from_num1, digit_from_num2, digit_from_num3)
            
            # Convert the smallest digit back to a string and append it to our builder list.
            key_string_builder.append(str(smallest_digit_at_pos_i))
            
        # Step 3: Form the final key string.
        # Join the collected digit strings to create the complete four-digit key string.
        # For example, if key_string_builder is ['0', '7', '7', '7'], this forms "0777".
        key_as_string = "".join(key_string_builder)
        
        # Step 4: Convert the key string to an integer.
        # This conversion automatically removes any leading zeros from the string.
        # For example, int("0777") becomes the integer 777.
        # Similarly, int("0000") becomes the integer 0.
        final_key_as_int = int(key_as_string)
        
        return final_key_as_int