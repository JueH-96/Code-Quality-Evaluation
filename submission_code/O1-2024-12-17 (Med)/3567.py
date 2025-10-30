class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Parse the date
        year = int(date[:4])
        month = int(date[5:7])
        day = int(date[8:10])
        
        # Convert each component to binary without leading '0b'
        binary_year = bin(year)[2:]
        binary_month = bin(month)[2:]
        binary_day = bin(day)[2:]
        
        # Construct and return the binary date format
        return f"{binary_year}-{binary_month}-{binary_day}"