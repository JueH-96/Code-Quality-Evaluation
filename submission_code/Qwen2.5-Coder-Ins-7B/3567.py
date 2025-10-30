class Solution:
    def convertDateToBinary(self, date: str) -> str:
        year, month, day = map(int, date.split('-'))
        binary_year = bin(year)[2:].zfill(12)
        binary_month = bin(month)[2:].zfill(4)
        binary_day = bin(day)[2:].zfill(5)
        return f"{binary_year}-{binary_month}-{binary_day}"