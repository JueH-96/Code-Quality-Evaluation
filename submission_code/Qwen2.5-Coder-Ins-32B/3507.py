class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def is_special(n):
            count = 0
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    count += 1
                    if i != n // i:
                        count += 1
                if count > 2:
                    return False
            return count == 2

        special_count = 0
        for num in range(l, r + 1):
            if is_special(num):
                special_count += 1

        return r - l + 1 - special_count