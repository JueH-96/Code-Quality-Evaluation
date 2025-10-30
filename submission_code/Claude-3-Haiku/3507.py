class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        count = 0
        for x in range(l, r+1):
            divisors = set()
            for i in range(1, int(x**0.5) + 1):
                if x % i == 0:
                    divisors.add(i)
                    divisors.add(x // i)
            if len(divisors) != 2:
                count += 1
        return count