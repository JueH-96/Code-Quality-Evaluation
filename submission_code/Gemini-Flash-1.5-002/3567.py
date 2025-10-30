class Solution:
    def convertDateToBinary(self, date: str) -> str:
        year, month, day = map(int, date.split('-'))
        
        def to_binary(n):
            if n == 0:
                return "0"
            result = ""
            while n > 0:
                result = str(n % 2) + result
                n //= 2
            return result

        binary_year = to_binary(year)
        binary_month = to_binary(month)
        binary_day = to_binary(day)

        return f"{binary_year}-{binary_month}-{binary_day}"