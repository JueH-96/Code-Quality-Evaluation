class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the date into year, month, and day
        parts = date.split('-')
        
        # Convert each part to integer
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        
        # Convert each integer to binary without '0b' prefix
        year_bin = bin(year)[2:]
        month_bin = bin(month)[2:]
        day_bin = bin(day)[2:]
        
        # Join the binary strings with dashes
        binary_date = '-'.join([year_bin, month_bin, day_bin])
        
        return binary_date