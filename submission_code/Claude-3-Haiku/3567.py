class Solution:
    def convertDateToBinary(self, date: str) -> str:
        year, month, day = map(int, date.split('-'))
        
        binary_year = bin(year)[2:]
        binary_month = bin(month)[2:]
        binary_day = bin(day)[2:]
        
        return f"{binary_year}-{binary_month.zfill(2)}-{binary_day.zfill(2)}"