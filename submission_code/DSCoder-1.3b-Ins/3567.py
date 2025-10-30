class Solution:
    def convertDateToBinary(self, date: str) -> str:
        year, month, day = map(int, date.split('-'))
        year_binary = bin(year + 946681760)[2:]
        month_binary = bin(month + 100000)[2:]
        day_binary = bin(day + 100)[2:]
        return year_binary + '-' + month_binary + '-' + day_binary