class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        def is_beautiful(n):
            s = str(n)
            product = 1
            sum_d = 0
            for c in s:
                if c == '0':
                    return True
                d = int(c)
                product *= d
                sum_d += d
            return product % sum_d == 0
        
        count = 0
        for num in range(l, r + 1):
            if is_beautiful(num):
                count += 1
        return count