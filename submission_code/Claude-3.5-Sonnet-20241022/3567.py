class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split date into year, month, day
        year, month, day = date.split('-')
        
        # Convert each part to integer and then to binary without '0b' prefix
        year_bin = bin(int(year))[2:]
        month_bin = bin(int(month))[2:]
        day_bin = bin(int(day))[2:]
        
        # Join with hyphens
        return f"{year_bin}-{month_bin}-{day_bin}"