class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        def is_symmetric(x):
            s = str(x)
            n = len(s)
            if n % 2 != 0:
                return False
            half = n // 2
            return sum(int(s[i]) for i in range(half)) == sum(int(s[i]) for i in range(half, n))
        
        count = 0
        for num in range(low, high + 1):
            if is_symmetric(num):
                count += 1
        return count