class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the date string into year, month, and day
        year, month, day = date.split('-')
        
        # Convert each part to integer
        year_int = int(year)
        month_int = int(month)
        day_int = int(day)
        
        # Convert to binary representation without leading zeros
        # bin() returns a string with '0b' prefix, so we slice it off
        year_binary = bin(year_int)[2:]
        month_binary = bin(month_int)[2:]
        day_binary = bin(day_int)[2:]
        
        # Return the parts in the required format
        return f"{year_binary}-{month_binary}-{day_binary}"