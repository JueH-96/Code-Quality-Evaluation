class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the date string by '-'
        year, month, day = date.split('-')
        
        # Convert each part to integer and then to binary
        # bin() returns string like '0b101', so we slice from index 2
        binary_year = bin(int(year))[2:]
        binary_month = bin(int(month))[2:]
        binary_day = bin(int(day))[2:]
        
        # Join with '-' separator
        return f"{binary_year}-{binary_month}-{binary_day}"