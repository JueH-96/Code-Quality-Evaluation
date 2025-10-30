from typing import List

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        # make prices 1-indexed
        p = [0] + prices
        
        # dp[i][r] = min cost to get fruits i..n, given that we can take for free
        # any fruit j with i <= j <= r (r is the farthest index covered by freebies).
        # i in [1..n+1], r in [0..n]
        INF = 10**18
        dp = [[INF] * (n+1) for _ in range(n+2)]
        
        # base case: if i>n, no cost
        for r in range(n+1):
            dp[n+1][r] = 0
        
        # fill dp bottom-up
        for i in range(n, 0, -1):
            for r in range(n+1):
                # if fruit i is within free coverage
                if i <= r:
                    # take it for free
                    dp[i][r] = min(dp[i][r], dp[i+1][r])
                
                # option to purchase fruit i
                new_r = max(r, min(n, i + i))  # extend free coverage to i+1..i+i
                cost_if_buy = p[i] + dp[i+1][new_r]
                dp[i][r] = min(dp[i][r], cost_if_buy)
        
        # answer: starting at fruit 1 with no freebies (r=0)
        return dp[1][0]


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumCoins([3,1,2]))  # expected 4
    print(sol.minimumCoins([1,10,1,1]))  # expected 2