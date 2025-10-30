class Solution:
    def convertDateToBinary(self, date: str) -> str:
        year, month, day = date.split('-')
        binary_year = bin(int(year))[2:]  # Convert year to binary and remove '0b'
        binary_month = bin(int(month))[2:]  # Convert month to binary and remove '0b'
        binary_day = bin(int(day))[2:]  # Convert day to binary and remove '0b'
        
        return f"{binary_year}-{binary_month}-{binary_day}"