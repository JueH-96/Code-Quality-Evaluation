class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        
        for i in range(n-1, -1, -1):
            bit = 1 << i
            if (a & bit) == 0 and (b & bit) == 0:
                a |= bit
            elif (a & bit) and (b & bit):
                if a > b:
                    b |= bit
                else:
                    a |= bit
            else:
                if a < b:
                    a |= bit
                else:
                    b |= bit
        
        return (a * b) % MOD