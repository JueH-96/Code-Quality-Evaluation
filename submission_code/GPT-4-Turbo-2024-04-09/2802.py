class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(i, square_str):
            target = i
            length = len(square_str)
            
            # Dynamic programming array to check if a sum is possible
            dp = [False] * (target + 1)
            dp[0] = True
            
            # Check all substrings
            for start in range(length):
                if dp[target]:
                    break
                for end in range(start, length):
                    num = int(square_str[start:end+1])
                    for k in range(target, num - 1, -1):
                        if dp[k - num]:
                            dp[k] = True
            
            return dp[target]
        
        total_sum = 0
        for i in range(1, n + 1):
            square = i * i
            square_str = str(square)
            if can_partition(i, square_str):
                total_sum += square
        
        return total_sum