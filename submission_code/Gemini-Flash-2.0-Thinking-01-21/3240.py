class Solution:
    def get_price(self, num: int, x: int) -> int:
        binary_representation = bin(num)[2:]
        price = 0
        for i in range(len(binary_representation)):
            index = len(binary_representation) - i # 1-indexed from right
            if index % x == 0 and binary_representation[i] == '1':
                price += 1
        return price
        
    def get_sum_prices(self, num: int, x: int) -> int:
        total_price = 0
        for i in range(1, num + 1):
            total_price += self.get_price(i, x)
        return total_price
        
    def calculate_price_contribution(self, num: int, j: int) -> int:
        if j <= 0:
            return 0
        if 2**(j-1) > num:
            return 0
        quotient = num // (2**j)
        remainder = num % (2**j)
        contribution = quotient * (2**(j-1))
        if remainder >= 2**(j-1):
            contribution += (remainder - 2**(j-1) + 1)
        return contribution
        
    def calculate_sum_prices_efficiently(self, num: int, x: int) -> int:
        total_price = 0
        j = x
        while True:
            if 2**j > num:
                break
            total_price += self.calculate_price_contribution(num, j)
            j += x
        return total_price

    def findMaximumNumber(self, k: int, x: int) -> int:
        low = 1
        high = k + 2 # Increased upper bound to k+2 for safety
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            sum_prices = self.calculate_sum_prices_efficiently(mid, x)
            if sum_prices <= k:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans