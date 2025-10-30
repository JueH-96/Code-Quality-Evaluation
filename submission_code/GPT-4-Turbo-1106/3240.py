class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def price_of_num(num):
            return bin(num).count('1', -x)
        
        left, right = 1, k
        while left < right:
            mid = (left + right + 1) // 2
            total_price = sum(price_of_num(i) for i in range(1, mid + 1))
            if total_price <= k:
                left = mid
            else:
                right = mid - 1
        return left