class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0
        for x in range(low, high + 1):
            s = str(x)
            length = len(s)
            if length % 2 != 0:
                continue
            mid = length // 2
            first_part = s[:mid]
            second_part = s[mid:]
            if sum(int(c) for c in first_part) == sum(int(c) for c in second_part):
                count += 1
        return count