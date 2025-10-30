class Solution:
    def convertDateToBinary(self, date: str) -> str:
        parts = date.split('-')
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        year_bin = bin(year)[2:]
        month_bin = bin(month)[2:]
        day_bin = bin(day)[2:]
        return f"{year_bin}-{month_bin}-{day_bin}"