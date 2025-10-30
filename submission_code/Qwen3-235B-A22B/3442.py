class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        unique = []
        for x in rewardValues:
            if not unique or x != unique[-1]:
                unique.append(x)
        
        if not unique:
            return 0
        
        sum_total = sum(unique)
        dp_size = sum_total + 1
        dp = [False] * dp_size
        dp[0] = True
        current_max = 0
        
        for a in unique:
            upper = min(current_max, a - 1)
            # Iterate from upper down to 0
            for x in range(upper, -1, -1):
                if dp[x]:
                    new_x = x + a
                    if new_x < dp_size and not dp[new_x]:
                        dp[new_x] = True
                        if new_x > current_max:
                            current_max = new_x
        
        # Find the maximum x in dp
        for x in range(sum_total, -1, -1):
            if dp[x]:
                return x
        return 0