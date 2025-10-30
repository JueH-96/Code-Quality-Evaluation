class Solution:
    def punishmentNumber(self, n: int) -> int:
        def is_punishment_number(num: int, target: int) -> bool:
            num_str = str(num)
            if int(num_str) == target:
                return True
            for i in range(1, len(num_str)):
                if is_punishment_number(int(num_str[i:]), target - int(num_str[:i])):
                    return True
            return False

        punishment_sum = 0
        for i in range(1, n + 1):
            if is_punishment_number(i * i, i):
                punishment_sum += i * i
        return punishment_sum