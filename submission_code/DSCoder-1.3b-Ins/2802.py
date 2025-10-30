class Solution:
    def punishmentNumber(self, n: int) -> int:
        def is_part(num):
            i = 1
            while i * i <= num:
                if i * (i * i - num) == num:
                    return True
                i += 1
            return False

        res = 0
        for i in range(1, n + 1):
            if is_part(i):
                res += i * i
            else:
                res += i * (i + 1) * (i + 1)
        return res