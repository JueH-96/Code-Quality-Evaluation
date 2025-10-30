class Solution:
    def convertDateToBinary(self, date: str) -> str:
        parts = date.split('-')
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        bin_year = bin(year)[2:]
        bin_month = bin(month)[2:]
        bin_day = bin(day)[2:]
        return f"{bin_year}-{bin_month}-{bin_day}"