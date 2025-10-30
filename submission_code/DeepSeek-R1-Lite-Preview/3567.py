class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the date into year, month, and day
        parts = date.split('-')
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        
        # Convert each part to binary without '0b' prefix
        binary_year = bin(year)[2:]
        binary_month = bin(month)[2:]
        binary_day = bin(day)[2:]
        
        # Combine them with dashes
        result = f"{binary_year}-{binary_month}-{binary_day}"
        
        return result