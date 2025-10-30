class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        def is_symmetric(n):
            s = str(n)
            if len(s) % 2 == 1:
                return False
            mid = len(s) // 2
            return sum(int(x) for x in s[:mid]) == sum(int(x) for x in s[mid:])

        return sum(is_symmetric(n) for n in range(low, high + 1))