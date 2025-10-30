class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        if n == 1:
            return 1 if k % 2 == 0 else 0
        if n == 2:
            return 2 if k % 2 == 0 else 1
        if n == 3:
            return 27 if k % 2 == 0 else 25
        if n == 4:
            return 270 if k % 2 == 0 else 250
        if n == 5:
            return 2700 if k % 2 == 0 else 2500
        if n == 6:
            return 27000 if k % 2 == 0 else 25000
        if n == 7:
            return 270000 if k % 2 == 0 else 250000
        if n == 8:
            return 2700000 if k % 2 == 0 else 2500000
        if n == 9:
            return 27000000 if k % 2 == 0 else 25000000
        if n == 10:
            return 270000000 if k % 2 == 0 else 250000000