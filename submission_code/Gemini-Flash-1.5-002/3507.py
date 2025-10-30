class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        count = 0
        for i in range(l, r + 1):
            divisors = []
            for j in range(1, int(i**0.5) + 1):
                if i % j == 0:
                    divisors.append(j)
                    if j * j != i and j != 1:
                        divisors.append(i // j)
            if len(divisors) != 2:
                count +=1

        return count