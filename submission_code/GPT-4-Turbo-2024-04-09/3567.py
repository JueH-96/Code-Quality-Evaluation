class Solution:
    def convertDateToBinary(self, date: str) -> str:
        year, month, day = date.split('-')
        year_bin = bin(int(year))[2:]  # Convert year to binary and remove '0b'
        month_bin = bin(int(month))[2:]  # Convert month to binary and remove '0b'
        day_bin = bin(int(day))[2:]  # Convert day to binary and remove '0b'
        return f"{year_bin}-{month_bin}-{day_bin}"