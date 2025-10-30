class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for x in range(low, high + 1):
            s = str(x)
            n = len(s)
            if n % 2 != 0:
                continue
            half = n // 2
            sum_first = sum(int(c) for c in s[:half])
            sum_second = sum(int(c) for c in s[half:])
            if sum_first == sum_second:
                count += 1
        return count