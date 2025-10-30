class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        """
        Generates a key from three positive integers based on a specific digit-wise rule.

        The process involves:
        1. Padding each number with leading zeros to make it a four-digit string.
        2. For each digit position (1st to 4th), finding the minimum digit among the 
           three numbers.
        3. Concatenating these minimum digits to form a four-digit key string.
        4. Converting the key string to an integer, which removes any leading zeros.
        """
        
        # Step 1: Pad each number to a 4-digit string with leading zeros.
        # f-string formatting provides a concise way to do this.
        # e.g., f"{1:04d}" results in "0001".
        s1 = f"{num1:04d}"
        s2 = f"{num2:04d}"
        s3 = f"{num3:04d}"

        # Step 2: Generate the key's digits using a list comprehension.
        # - zip(s1, s2, s3) iterates through the strings, yielding tuples of 
        #   corresponding digits, e.g., (s1[0], s2[0], s3[0]), then (s1[1], s2[1], s3[1]), etc.
        # - For each tuple of digit characters, min() finds the smallest one.
        #   This works correctly as character comparison ('0'<'1'<...<'9') matches numerical order.
        # The result is a list of characters representing the digits of the key.
        key_digits = [min(d1, d2, d3) for d1, d2, d3 in zip(s1, s2, s3)]

        # Step 3: Combine the digits into a string and convert to an integer.
        # - "".join(key_digits) concatenates the characters into a single string.
        # - int() converts this string to an integer, automatically handling
        #   leading zeros (e.g., int("0777") becomes 777).
        return int("".join(key_digits))