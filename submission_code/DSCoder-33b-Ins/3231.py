class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        curr_sum = 0
        count = 0
        for coin in coins:
            if coin > target:
                break
            if coin > curr_sum + 1:
                return count
            curr_sum += coin
            count += 1
        while curr_sum < target:
            curr_sum += curr_sum + 1
            count += 1
        return count