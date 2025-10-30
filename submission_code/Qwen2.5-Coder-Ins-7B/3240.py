class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_set_bits(num):
            return bin(num).count('1')
        
        def price(num):
            return sum(count_set_bits(num) for i in range(1, num + 1) if i % x == 0)
        
        left, right = 1, k
        while left <= right:
            mid = (left + right) // 2
            if price(mid) <= k:
                left = mid + 1
            else:
                right = mid - 1
        return right