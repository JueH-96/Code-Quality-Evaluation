class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for x in range(low, high + 1):
            s = str(x)
            n = len(s)
            if n % 2 != 0:
                continue
            mid = n // 2
            sum1 = sum(int(d) for d in s[:mid])
            sum2 = sum(int(d) for d in s[mid:])
            if sum1 == sum2:
                count += 1
        return count