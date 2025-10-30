class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7

        def countSetBits(n):
            count = 0
            while n > 0:
                count += n & 1
                n >>= 1
            return count

        def isKReducible(n, k):
            for _ in range(k):
                if n == 1:
                    return True
                n = countSetBits(n)
            return n == 1

        n = int(s, 2)
        count = 0
        for i in range(1, n):
            if isKReducible(i, k):
                count += 1
        return count