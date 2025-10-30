class Solution:
    def convertDateToBinary(self, date: str) -> str:
        year_str, month_str, day_str = date.split('-')
        year = int(year_str)
        month = int(month_str)
        day = int(day_str)
        year_bin = bin(year)[2:]
        month_bin = bin(month)[2:]
        day_bin = bin(day)[2:]
        return f"{year_bin}-{month_bin}-{day_bin}"