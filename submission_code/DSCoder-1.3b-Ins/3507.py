class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        count = 0
        for i in range(l, r+1):
            divisors = [1]
            for j in range(2, int(i**0.5)+1):
                if i % j == 0:
                    if i // j == j:
                        divisors.append(j)
                    else:
                        divisors.extend([j, i // j])
            if len(divisors) == 2:
                count += 1
        return count