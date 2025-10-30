class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the date into year, month, and day
        year, month, day = date.split('-')
        
        # Convert each part to integer, then to binary (removing '0b' prefix)
        year_binary = bin(int(year))[2:]
        month_binary = bin(int(month))[2:]
        day_binary = bin(int(day))[2:]
        
        # Join them with '-'
        return f"{year_binary}-{month_binary}-{day_binary}"