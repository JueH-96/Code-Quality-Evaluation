class Solution:
    def punishmentNumber(self, n: int) -> int:
        total = 0
        for i in range(1, n + 1):
            square = i * i
            s = str(square)
            if self.check_split(s, 0, 0, i):
                total += square
        return total

    def check_split(self, s, pos, current_sum, target):
        if current_sum > target:
            return False
        if pos == len(s):
            return current_sum == target
        for i in range(pos + 1, len(s) + 1):
            substring = s[pos:i]
            if len(substring) > 1 and substring[0] == '0':
                continue
            num = int(substring)
            if current_sum + num > target:
                break
            if self.check_split(s, i, current_sum + num, target):
                return True
        return False