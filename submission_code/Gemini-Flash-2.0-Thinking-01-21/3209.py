class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        memo = {}
        
        def solve(index, free_until):
            if index > n:
                return 0
            if index <= free_until:
                return solve(index + 1, free_until)
            if (index, free_until) in memo:
                return memo[(index, free_until)]
            
            buy_cost = prices[index-1] + solve(index + 1, max(free_until, index + index))
            
            result = buy_cost
            memo[(index, free_until)] = result
            return result
            
        return solve(1, 0)