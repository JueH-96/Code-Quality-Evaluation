class Solution:
    def punishmentNumber(self, n: int) -> int:
        def is_punishment(num):
            square = str(num * num)
            length = len(square)
            for mask in range(1 << length - 1):
                total, cur = 0, 0
                for i in range(length):
                    cur = cur * 10 + int(square[i])
                    if mask & (1 << i):
                        total += cur
                        cur = 0
                total += cur
                if total == num:
                    return True
            return False

        res = 0
        for i in range(1, n + 1):
            if is_punishment(i):
                res += i * i
        return res