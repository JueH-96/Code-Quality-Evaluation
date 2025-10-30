class Solution:
    def punishmentNumber(self, n: int) -> int:
        def is_partition_sum_possible(s, target_sum):
            def check_partition(start_index, remaining_sum):
                if start_index == len(s):
                    return remaining_sum == 0
                if remaining_sum < 0:
                    return False
                for end_index in range(start_index, len(s)):
                    substring = s[start_index:end_index+1]
                    value = int(substring)
                    if check_partition(end_index + 1, remaining_sum - value):
                        return True
                return False
            return check_partition(0, target_sum)
            
        punishment_sum = 0
        for i in range(1, n + 1):
            square = i * i
            square_str = str(square)
            if is_partition_sum_possible(square_str, i):
                punishment_sum += square
        return punishment_sum