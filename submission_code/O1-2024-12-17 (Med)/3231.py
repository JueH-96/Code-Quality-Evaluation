class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        coverage = 1
        added_coins = 0
        i = 0
        
        while coverage <= target:
            # If we can use the existing coin to extend coverage
            if i < len(coins) and coins[i] <= coverage:
                coverage += coins[i]
                i += 1
            else:
                # We must add a coin of value equal to 'coverage'
                coverage += coverage
                added_coins += 1
                
        return added_coins