class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for num in range(low, high + 1):
            s = str(num)
            length = len(s)
            if length % 2 != 0:
                continue
            n = length // 2
            first_sum = sum(int(c) for c in s[:n])
            second_sum = sum(int(c) for c in s[n:])
            if first_sum == second_sum:
                count += 1
        return count