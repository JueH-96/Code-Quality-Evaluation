class Solution:
    def punishmentNumber(self, n: int) -> int:
        def is_punishment_number(i):
            square_str = str(i * i)
            
            def dfs(index, current_sum):
                if index == len(square_str):
                    return current_sum == i
                for j in range(index, len(square_str)):
                    num = int(square_str[index:j+1])
                    if dfs(j + 1, current_sum + num):
                        return True
                return False
            
            return dfs(0, 0)
        
        return sum(i * i for i in range(1, n + 1) if is_punishment_number(i))