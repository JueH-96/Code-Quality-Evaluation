class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(square_str, target_sum):
            def solve(start_index, current_sum):
                if start_index == len(square_str):
                    return current_sum == 0
                for end_index in range(start_index, len(square_str)):
                    substring = square_str[start_index:end_index+1]
                    value = int(substring)
                    if solve(end_index + 1, current_sum - value):
                        return True
                return False
            return solve(0, target_sum)

        punishment_sum = 0
        for i in range(1, n + 1):
            square = i * i
            square_str = str(square)
            if can_partition(square_str, i):
                punishment_sum += square
        return punishment_sum