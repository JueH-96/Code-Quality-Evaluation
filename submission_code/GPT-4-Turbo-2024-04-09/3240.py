class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def price_of_num(num):
            binary = bin(num)[2:]  # Get binary representation without '0b' prefix
            return sum(1 for i in range(len(binary)) if (i + 1) % x == 0 and binary[-(i + 1)] == '1')

        total_price = 0
        num = 0
        while True:
            num += 1
            total_price += price_of_num(num)
            if total_price > k:
                return num - 1