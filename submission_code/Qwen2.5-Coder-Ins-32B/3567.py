class Solution:
    def convertDateToBinary(self, date: str) -> str:
        year, month, day = date.split('-')
        return f"{int(year):b}-{int(month):b}-{int(day):b}"