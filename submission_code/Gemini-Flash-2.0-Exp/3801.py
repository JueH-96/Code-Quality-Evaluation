class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        def is_beautiful(n):
            s = 0
            p = 1
            temp = n
            while temp > 0:
                digit = temp % 10
                s += digit
                p *= digit
                temp //= 10
            if s == 0:
                return False
            if p % s == 0:
                return True
            else:
                return False

        count = 0
        for i in range(l, r + 1):
            if is_beautiful(i):
                count += 1
        return count