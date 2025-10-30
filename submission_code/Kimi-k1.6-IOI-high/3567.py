class Solution:
    def convertDateToBinary(self, date: str) -> str:
        parts = date.split('-')
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        binary_year = bin(year)[2:]
        binary_month = bin(month)[2:]
        binary_day = bin(day)[2:]
        return f"{binary_year}-{binary_month}-{binary_day}"