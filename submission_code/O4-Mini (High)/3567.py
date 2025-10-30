class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the date into year, month, and day parts
        year_str, month_str, day_str = date.split('-')
        
        # Convert each part to integer, then to binary string without '0b' prefix
        year_bin = bin(int(year_str))[2:]
        month_bin = bin(int(month_str))[2:]
        day_bin = bin(int(day_str))[2:]
        
        # Join the binary parts with hyphens
        return f"{year_bin}-{month_bin}-{day_bin}"