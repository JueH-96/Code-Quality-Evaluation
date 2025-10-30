class Solution:
    def punishmentNumber(self, n: int) -> int:
        total = 0
        for i in range(1, n + 1):
            square = i * i
            s = str(square)
            if self.can_partition(s, i):
                total += square
        return total

    def can_partition(self, s: str, target: int) -> bool:
        def helper(index, current_sum):
            if index == len(s):
                return current_sum == target
            for end in range(index + 1, len(s) + 1):
                num = int(s[index:end])
                if helper(end, current_sum + num):
                    return True
            return False
        return helper(0, 0)