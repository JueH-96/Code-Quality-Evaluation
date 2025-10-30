class Solution:
    def convertDateToBinary(self, date: str) -> str:
        year, month, day = map(int, date.split('-'))
        return f"{bin(year)[2:]}-{bin(month)[2:].zfill(4)}-{bin(day)[2:].zfill(5)}"