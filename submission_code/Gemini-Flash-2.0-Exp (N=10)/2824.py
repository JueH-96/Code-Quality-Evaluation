class Solution:
    def isFascinating(self, n: int) -> bool:
        s = str(n) + str(2 * n) + str(3 * n)
        if len(s) != 9:
            return False
        counts = {}
        for digit in s:
            if digit == '0':
                return False
            counts[digit] = counts.get(digit, 0) + 1
        for i in range(1, 10):
            if str(i) not in counts or counts[str(i)] != 1:
                return False
        return True