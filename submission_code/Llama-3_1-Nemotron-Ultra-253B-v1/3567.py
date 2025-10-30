class Solution:
    def convertDateToBinary(self, date: str) -> str:
        year, month, day = date.split('-')
        bin_year = bin(int(year))[2:]
        bin_month = bin(int(month))[2:]
        bin_day = bin(int(day))[2:]
        return f"{bin_year}-{bin_month}-{bin_day}"