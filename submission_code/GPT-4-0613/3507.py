class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def is_special(n):
            count = 0
            for i in range(1, int(n**0.5)+1):
                if n % i == 0:
                    if n / i == i:
                        count += 1
                    else:
                        count += 2
                if count > 3:
                    return False
            return count == 3

        count = 0
        for i in range(l, r+1):
            if not is_special(i):
                count += 1
        return count