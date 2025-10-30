class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def calculate_price(num, x):
            binary = bin(num)[2:]
            price = 0
            for i in range(len(binary)):
                if (len(binary) - i) % x == 0 and binary[i] == '1':
                    price += 1
            return price

        def calculate_total_price(num, x):
            total_price = 0
            for i in range(1, num + 1):
                total_price += calculate_price(i, x)
            return total_price

        low = 0
        high = 2 * 10**16
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            total_price = calculate_total_price(mid, x)
            if total_price <= k:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans