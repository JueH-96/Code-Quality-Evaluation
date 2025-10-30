class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for i in range(low, high + 1):
            s = str(i)
            n = len(s)
            if n % 2 == 0:
                mid = n // 2
                sum1 = 0
                for j in range(mid):
                    sum1 += int(s[j])
                sum2 = 0
                for j in range(mid, n):
                    sum2 += int(s[j])
                if sum1 == sum2:
                    count += 1
        return count