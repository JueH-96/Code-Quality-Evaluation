class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        total_price = 0
        num = 0
        while total_price <= k:
            num += 1
            price = 0
            binary = bin(num)[2:]
            for i in range(len(binary)):
                if (i + 1) % x == 0 and binary[len(binary) - 1 - i] == '1':
                    price += 1
            total_price += price
        return num -1