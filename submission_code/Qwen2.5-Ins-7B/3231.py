class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        added_coins = 0
        current_sum = 0
        
        for coin in coins:
            while coin > current_sum + 1 and current_sum < target:
                added_coins += 1
                current_sum += current_sum + 1
            if coin <= current_sum + 1:
                current_sum += coin
                if current_sum >= target:
                    return added_coins
        while current_sum < target:
            added_coins += 1
            current_sum += current_sum + 1
        return added_coins