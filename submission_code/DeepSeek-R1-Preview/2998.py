class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for x in range(low, high + 1):
            s = str(x)
            if len(s) % 2 != 0:
                continue
            n = len(s) // 2
            sum1 = sum(int(c) for c in s[:n])
            sum2 = sum(int(c) for c in s[n:])
            if sum1 == sum2:
                count += 1
        return count