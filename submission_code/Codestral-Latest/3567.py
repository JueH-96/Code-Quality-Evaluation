class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the date into year, month, and day
        year, month, day = date.split('-')

        # Convert year, month, and day to integers
        year_int = int(year)
        month_int = int(month)
        day_int = int(day)

        # Convert year, month, and day to their binary representations without leading zeroes
        year_bin = bin(year_int)[2:]
        month_bin = bin(month_int)[2:]
        day_bin = bin(day_int)[2:]

        # Join the binary representations in the format year-month-day
        binary_date = f"{year_bin}-{month_bin}-{day_bin}"

        return binary_date