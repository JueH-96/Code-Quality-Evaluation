class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Step 1: Parse the date string "yyyy-mm-dd"
        # Example: "2080-02-29" -> ["2080", "02", "29"]
        year_str, month_str, day_str = date.split('-')

        # Step 2: Convert string parts to integers
        # Example: "2080" -> 2080, "02" -> 2, "29" -> 29
        # The int() function correctly handles leading zeros in the string representation,
        # e.g., int("02") becomes 2.
        year_int = int(year_str)
        month_int = int(month_str)
        day_int = int(day_str)

        # Step 3: Convert integers to binary strings (without "0b" prefix)
        # Python's bin() function converts an integer to its binary representation.
        # It prepends "0b" to the binary string (e.g., bin(5) results in "0b101").
        # We remove this prefix by slicing the string from index 2 (i.e., bin(num)[2:]).
        # The problem asks for binary representation "without any leading zeroes".
        # For positive integers (which year, month, and day will be),
        # bin(num)[2:] correctly produces this. For example, bin(1)[2:] is "1".
        
        # Convert year to its binary string representation
        # Example: 2080 -> bin(2080) is "0b100000100000" -> bin(2080)[2:] is "100000100000"
        year_bin = bin(year_int)[2:]
        
        # Convert month to its binary string representation
        # Example: 2 -> bin(2) is "0b10" -> bin(2)[2:] is "10"
        month_bin = bin(month_int)[2:]
        
        # Convert day to its binary string representation
        # Example: 29 -> bin(29) is "0b11101" -> bin(29)[2:] is "11101"
        day_bin = bin(day_int)[2:]

        # Step 4: Concatenate the binary strings with hyphens
        # The final format required is "binary_year-binary_month-binary_day".
        # An f-string provides a clean way to format this.
        # Example: "100000100000" + "-" + "10" + "-" + "11101"
        # Result: "100000100000-10-11101"
        binary_date_representation = f"{year_bin}-{month_bin}-{day_bin}"
        
        return binary_date_representation