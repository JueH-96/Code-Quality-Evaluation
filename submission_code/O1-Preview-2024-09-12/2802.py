class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(s_str, target):
            def dfs(start_idx, current_sum):
                if start_idx == len(s_str):
                    return current_sum == target
                for end_idx in range(start_idx + 1, len(s_str) + 1):
                    substring = s_str[start_idx:end_idx]
                    value = int(substring)
                    if dfs(end_idx, current_sum + value):
                        return True
                return False
            return dfs(0, 0)
        
        total = 0
        for i in range(1, n + 1):
            square = i * i
            s_str = str(square)
            if can_partition(s_str, i):
                total += square
        return total