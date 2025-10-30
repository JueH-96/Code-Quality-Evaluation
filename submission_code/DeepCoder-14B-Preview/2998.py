class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for x in range(low, high + 1):
            s = str(x)
            if len(s) % 2 != 0:
                continue
            n = len(s) // 2
            first_half = s[:n]
            second_half = s[n:]
            sum_first = sum(int(c) for c in first_half)
            sum_second = sum(int(c) for c in second_half)
            if sum_first == sum_second:
                count += 1
        return count