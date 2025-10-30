class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Sort offers based on the end index
        offers.sort(key=lambda x: x[1])
        
        # Initialize dp array where dp[i] is the maximum gold we can get by considering offers up to i
        dp = [0] * len(offers)
        
        # Function to find the last offer that ends before the current offer starts
        def find_last_non_conflicting_offer(current_index):
            low, high = 0, current_index - 1
            while low <= high:
                mid = (low + high) // 2
                if offers[mid][1] < offers[current_index][0]:
                    if offers[mid + 1][1] < offers[current_index][0]:
                        low = mid + 1
                    else:
                        return mid
                else:
                    high = mid - 1
            return -1
        
        # Fill the dp array
        for i in range(len(offers)):
            # Profit including the current offer
            current_profit = offers[i][2]
            last_non_conflicting_index = find_last_non_conflicting_offer(i)
            if last_non_conflicting_index != -1:
                current_profit += dp[last_non_conflicting_index]
            
            # Maximum profit by either including or excluding the current offer
            if i == 0:
                dp[i] = current_profit
            else:
                dp[i] = max(dp[i - 1], current_profit)
        
        # The answer is the maximum profit we can get by considering all offers
        return dp[-1]