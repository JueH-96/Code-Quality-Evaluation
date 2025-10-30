class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def calculate_price(num: int, x: int) -> int:
            """Calculate the price of a number."""
            price = 0
            binary = bin(num)[2:][::-1]  # Reverse binary representation
            for i in range(len(binary)):
                if (i + 1) % x == 0 and binary[i] == '1':
                    price += 1
            return price

        def calculate_sum_price(num: int, x: int) -> int:
            """Calculate the sum of prices of all numbers from 1 to num."""
            sum_price = 0
            for i in range(1, num + 1):
                sum_price += calculate_price(i, x)
            return sum_price

        left, right = 1, 10**18
        while left < right:
            mid = (left + right + 1) // 2
            if calculate_sum_price(mid, x) <= k:
                left = mid
            else:
                right = mid - 1
        return left