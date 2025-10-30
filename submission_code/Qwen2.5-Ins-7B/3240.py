class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_set_bits(num, pos):
            count = 0
            for i in range(pos - 1, 60, pos):
                if num & (1 << i):
                    count += 1
            return count
        
        left, right = 1, 10**15
        while left < right:
            mid = (left + right + 1) // 2
            price = 0
            for pos in range(x, 61, x):
                price += count_set_bits(mid, pos)
            if price > k:
                right = mid - 1
            else:
                left = mid
        return left