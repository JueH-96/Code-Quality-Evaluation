class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the date into year, month, and day
        year, month, day = date.split('-')
        
        # Convert each part to an integer
        year = int(year)
        month = int(month)
        day = int(day)
        
        # Convert each part to binary without leading zeroes
        year_binary = bin(year)[2:]
        month_binary = bin(month)[2:]
        day_binary = bin(day)[2:]
        
        # Concatenate the binary representations with dashes
        return f"{year_binary}-{month_binary}-{day_binary}"