class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the date into year, month, and day components.
        year_str, month_str, day_str = date.split('-')
        
        # Convert each component to integer, then to binary string without the '0b' prefix.
        year_bin = bin(int(year_str))[2:]
        month_bin = bin(int(month_str))[2:]
        day_bin = bin(int(day_str))[2:]
        
        # Combine the binary representations with '-' in between.
        return f"{year_bin}-{month_bin}-{day_bin}"