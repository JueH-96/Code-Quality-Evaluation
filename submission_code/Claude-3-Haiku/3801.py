class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        count = 0
        for num in range(l, r+1):
            product = 1
            total = 0
            for digit in str(num):
                product *= int(digit)
                total += int(digit)
            if product % total == 0:
                count += 1
        return count