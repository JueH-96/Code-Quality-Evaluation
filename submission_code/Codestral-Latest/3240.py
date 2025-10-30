class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_set_bits(num, x):
            count = 0
            while num > 0:
                if num % 2 == 1:
                    count += 1
                num //= 2
            return count

        def price(num, x):
            s = bin(num)[2:]
            return sum(1 for i in range(len(s)) if (i + 1) % x == 0 and s[~i] == '1')

        num = 1
        total_price = 0

        while total_price <= k:
            total_price += price(num, x)
            if total_price > k:
                return num - 1
            num += 1

        return num - 1