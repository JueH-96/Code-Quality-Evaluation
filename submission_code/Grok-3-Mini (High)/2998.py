class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for num in range(low, high + 1):
            s = str(num)
            length = len(s)
            if length % 2 == 1:
                continue
            n = length // 2
            sum_first = sum(int(s[i]) for i in range(n))
            sum_last = sum(int(s[i + n]) for i in range(n))
            if sum_first == sum_last:
                count += 1
        return count