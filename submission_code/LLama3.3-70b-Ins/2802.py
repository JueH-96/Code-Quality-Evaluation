class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(num: int, target: int) -> bool:
            """Check if the decimal representation of num can be partitioned into contiguous substrings 
            such that the sum of the integer values of these substrings equals target."""
            num_str = str(num)
            def dfs(index: int, path_sum: int) -> bool:
                if index == len(num_str):
                    return path_sum == target
                for i in range(index, len(num_str)):
                    substr = num_str[index:i+1]
                    if int(substr) > target:
                        break
                    if dfs(i+1, path_sum + int(substr)):
                        return True
                return False
            return dfs(0, 0)

        punishment_number = 0
        for i in range(1, n+1):
            square = i * i
            if can_partition(square, i):
                punishment_number += square
        return punishment_number