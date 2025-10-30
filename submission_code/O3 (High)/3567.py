class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the input by '-'
        year_str, month_str, day_str = date.split('-')
        
        # Convert each part to integer
        year = int(year_str)
        month = int(month_str)
        day = int(day_str)
        
        # Convert each integer to binary (strip the leading "0b")
        year_bin = bin(year)[2:]
        month_bin = bin(month)[2:]
        day_bin = bin(day)[2:]
        
        # Join with hyphens and return
        return f"{year_bin}-{month_bin}-{day_bin}"