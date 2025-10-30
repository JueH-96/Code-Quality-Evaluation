class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the date string by hyphens
        year_str, month_str, day_str = date.split('-')
        
        # Convert strings to integers
        year_int = int(year_str)
        month_int = int(month_str)
        day_int = int(day_str)
        
        # Convert integers to binary strings without the '0b' prefix
        bin_year = bin(year_int)[2:]
        bin_month = bin(month_int)[2:]
        bin_day = bin(day_int)[2:]
        
        # Join the binary strings with hyphens and return the result
        result = bin_year + '-' + bin_month + '-' + bin_day
        return result