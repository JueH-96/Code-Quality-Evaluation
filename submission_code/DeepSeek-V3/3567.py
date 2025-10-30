class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the date into year, month, and day
        year, month, day = date.split('-')
        
        # Convert each part to integer
        year_int = int(year)
        month_int = int(month)
        day_int = int(day)
        
        # Convert each part to binary without leading zeros
        year_bin = bin(year_int)[2:]
        month_bin = bin(month_int)[2:]
        day_bin = bin(day_int)[2:]
        
        # Combine them into the required format
        binary_date = f"{year_bin}-{month_bin}-{day_bin}"
        
        return binary_date