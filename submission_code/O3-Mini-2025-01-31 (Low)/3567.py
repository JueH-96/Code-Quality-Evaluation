class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split date into year, month, day parts using the '-' delimiter
        year_str, month_str, day_str = date.split('-')
        # Convert each component to integer and then to binary string using bin(), slicing off the "0b"
        year_binary = bin(int(year_str))[2:]
        month_binary = bin(int(month_str))[2:]
        day_binary = bin(int(day_str))[2:]
        # Concatenate the binary strings in the format year-month-day
        return f"{year_binary}-{month_binary}-{day_binary}"