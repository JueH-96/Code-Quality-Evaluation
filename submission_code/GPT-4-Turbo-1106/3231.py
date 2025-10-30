class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        reach, added_coins = 0, 0
        
        for coin in coins:
            while coin > reach + 1 and reach < target:
                reach += reach + 1
                added_coins += 1
            reach += coin
            if reach >= target:
                break
        
        while reach < target:
            reach += reach + 1
            added_coins += 1
        
        return added_coins