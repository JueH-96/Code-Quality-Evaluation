class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the input date string into year, month, and day
        year_str, month_str, day_str = date.split('-')
        
        # Convert each part to an integer
        year_int = int(year_str)
        month_int = int(month_str)
        day_int = int(day_str)
        
        # Convert each integer to its binary representation (without '0b' prefix)
        year_bin = bin(year_int)[2:]
        month_bin = bin(month_int)[2:]
        day_bin = bin(day_int)[2:]
        
        # Construct and return the final string
        return f"{year_bin}-{month_bin}-{day_bin}"