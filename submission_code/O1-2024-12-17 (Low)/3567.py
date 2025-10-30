class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Extract year, month, and day from the date
        year_str = date[:4]
        month_str = date[5:7]
        day_str = date[8:10]

        # Convert them to integers
        year_int = int(year_str)
        month_int = int(month_str)
        day_int = int(day_str)

        # Convert each integer to binary and remove the '0b' prefix
        bin_year = bin(year_int)[2:]
        bin_month = bin(month_int)[2:]
        bin_day = bin(day_int)[2:]

        # Construct the result in yyyy-mm-dd format using binary strings
        return f"{bin_year}-{bin_month}-{bin_day}"