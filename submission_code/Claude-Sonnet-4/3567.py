class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the date string by '-'
        parts = date.split('-')
        
        # Convert each part to integer, then to binary (removing '0b' prefix)
        year_binary = bin(int(parts[0]))[2:]
        month_binary = bin(int(parts[1]))[2:]
        day_binary = bin(int(parts[2]))[2:]
        
        # Join with hyphens
        return f"{year_binary}-{month_binary}-{day_binary}"