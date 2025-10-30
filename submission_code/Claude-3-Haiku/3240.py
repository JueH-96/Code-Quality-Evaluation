class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        total_price = 0
        num = 0
        while total_price <= k:
            binary_rep = bin(num)[2:][::-1]
            price = sum(1 for i, bit in enumerate(binary_rep) if int(bit) == 1 and (i+1) % x == 0)
            total_price += price
            if total_price <= k:
                num += 1
            else:
                return num - 1
        return num - 1