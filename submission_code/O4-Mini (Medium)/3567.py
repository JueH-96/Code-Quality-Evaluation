class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the input string into year, month, and day
        year_str, month_str, day_str = date.split('-')
        
        # Convert each part to an integer, then to binary, and strip the "0b" prefix
        year_bin = bin(int(year_str))[2:]
        month_bin = bin(int(month_str))[2:]
        day_bin = bin(int(day_str))[2:]
        
        # Join the binary representations with hyphens and return
        return f"{year_bin}-{month_bin}-{day_bin}"