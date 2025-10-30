class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def price(num, x):
            binary = bin(num)[2:][::-1]
            return sum(1 for i in range(len(binary)) if i % x == 0 and binary[i] == '1')
        
        def sum_of_prices(n, x):
            return sum(price(i, x) for i in range(1, n + 1))
        
        left, right = 1, 10**15
        while left < right:
            mid = (left + right + 1) // 2
            if sum_of_prices(mid, x) <= k:
                left = mid
            else:
                right = mid - 1
        return left