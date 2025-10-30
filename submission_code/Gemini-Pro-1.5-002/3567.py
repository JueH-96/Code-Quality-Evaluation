class Solution:
    def convertDateToBinary(self, date: str) -> str:
        year, month, day = date.split('-')
        year = int(year)
        month = int(month)
        day = int(day)
        
        year_bin = bin(year)[2:]
        month_bin = bin(month)[2:]
        day_bin = bin(day)[2:]
        
        return year_bin + "-" + month_bin + "-" + day_bin