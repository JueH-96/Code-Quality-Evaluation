import bisect

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort the offers based on their end times
        offers_sorted = sorted(offers, key=lambda x: x[1])
        m = len(offers_sorted)
        dp = [0]  # dp[0] represents the maximum profit with 0 offers considered
        end_times = []
        
        for i in range(m):
            start = offers_sorted[i][0]
            end = offers_sorted[i][1]
            gold = offers_sorted[i][2]
            
            # Find the largest end time that is less than the current offer's start time
            idx = bisect.bisect_left(end_times, start)
            j = idx - 1
            
            if j >= 0:
                current_profit = dp[j + 1] + gold
            else:
                current_profit = gold
            
            # Update the DP array
            new_max = max(dp[-1], current_profit)
            dp.append(new_max)
            
            # Insert the current end time into the list while maintaining order
            bisect.insort(end_times, end)
        
        return dp[-1]