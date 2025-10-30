class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for num in range(low, high + 1):
            s = str(num)
            if len(s) % 2 != 0:
                continue
            n = len(s) // 2
            if sum(map(int, s[:n])) == sum(map(int, s[n:])):
                count += 1
        return count