class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        def is_symmetric(x):
            s = str(x)
            n = len(s)
            if n % 2 != 0:
                return False
            mid = n // 2
            return sum(int(d) for d in s[:mid]) == sum(int(d) for d in s[mid:])
        
        count = 0
        for x in range(low, high + 1):
            if is_symmetric(x):
                count += 1
        return count