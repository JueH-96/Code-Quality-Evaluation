class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Parse the date string
        year = int(date[0:4])
        month = int(date[5:7])
        day = int(date[8:10])
        
        # Convert year, month, and day to binary (removing the '0b' prefix)
        bin_year = bin(year)[2:]
        bin_month = bin(month)[2:]
        bin_day = bin(day)[2:]
        
        # Combine them in the specified format
        return f"{bin_year}-{bin_month}-{bin_day}"