class Solution:
    def convertDateToBinary(self, date: str) -> str:
        parts = date.split('-')
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])

        year_binary = bin(year)[2:]
        month_binary = bin(month)[2:]
        day_binary = bin(day)[2:]

        return f"{year_binary}-{month_binary}-{day_binary}"