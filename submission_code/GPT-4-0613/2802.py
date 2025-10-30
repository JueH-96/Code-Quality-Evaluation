class Solution:
    def punishmentNumber(self, n: int) -> int:
        def is_punishment_number(num):
            square = num * num
            str_square = str(square)
            for i in range(1, len(str_square)):
                for j in range(i, len(str_square)):
                    if num == int(str_square[:i]) + int(str_square[i:j+1]) + (int(str_square[j+1:]) if j+1 < len(str_square) else 0):
                        return True
            return False

        punishment_sum = 0
        for i in range(1, n+1):
            if is_punishment_number(i):
                punishment_sum += i * i
        return punishment_sum