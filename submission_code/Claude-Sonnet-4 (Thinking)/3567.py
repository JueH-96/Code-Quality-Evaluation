class Solution:
    def convertDateToBinary(self, date: str) -> str:
        year, month, day = date.split('-')
        
        year_int = int(year)
        month_int = int(month)
        day_int = int(day)
        
        year_bin = bin(year_int)[2:]  # Remove '0b' prefix
        month_bin = bin(month_int)[2:]  # Remove '0b' prefix
        day_bin = bin(day_int)[2:]  # Remove '0b' prefix
        
        return f"{year_bin}-{month_bin}-{day_bin}"