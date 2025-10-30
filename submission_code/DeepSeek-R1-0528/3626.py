class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        for num in range(n, 101):
            product = 1
            num_str = str(num)
            for char in num_str:
                digit = int(char)
                product *= digit
                if product == 0:
                    break
            if product % t == 0:
                return num