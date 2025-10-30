class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # 1. Split the date string into year, month, and day components
        # The date string is in "yyyy-mm-dd" format.
        parts = date.split('-')
        
        # 2. Convert each string part to its integer equivalent
        year_int = int(parts[0])
        month_int = int(parts[1])
        day_int = int(parts[2])
        
        # 3. Convert each integer to its binary string representation.
        # The bin() function returns a string prefixed with "0b" (e.g., "0b10").
        # We need to slice the string to remove this "0b" prefix to get
        # the pure binary representation without leading zeros.
        binary_year = bin(year_int)[2:]
        binary_month = bin(month_int)[2:]
        binary_day = bin(day_int)[2:]
        
        # 4. Join the binary strings with hyphens to form the final result.
        # An f-string provides a concise way to format the output.
        result = f"{binary_year}-{binary_month}-{binary_day}"
        
        return result