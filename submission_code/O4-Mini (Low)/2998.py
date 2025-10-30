class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for x in range(low, high + 1):
            s = str(x)
            L = len(s)
            if L % 2 == 1:
                continue
            n = L // 2
            if sum(int(ch) for ch in s[:n]) == sum(int(ch) for ch in s[n:]):
                count += 1
        return count