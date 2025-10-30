class Solution:
    def convertDateToBinary(self, date: str) -> str:
        # Split the input string into year, month, and day components
        year_str, month_str, day_str = date.split('-')
        
        # Convert each component to integer (this removes any leading zeros)
        year = int(year_str)
        month = int(month_str)
        day = int(day_str)
        
        # Convert each integer to its binary representation without leading zeroes
        bin_year = bin(year)[2:]
        bin_month = bin(month)[2:]
        bin_day = bin(day)[2:]
        
        # Return the concatenated result in the desired format
        return f"{bin_year}-{bin_month}-{bin_day}"