class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def price(num, x):
            p = 0
            for i in range(1, num.bit_length() + 1):
                if i % x == 0:
                    p += (num >> i) & 1
            return p
        
        low, high = 1, 10**15
        while low < high:
            mid = (low + high + 1) // 2
            if sum(price(i, x) for i in range(1, mid + 1)) <= k:
                low = mid
            else:
                high = mid - 1
        return low