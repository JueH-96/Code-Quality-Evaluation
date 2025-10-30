class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Create a list to store the number of coins at each position
        max_pos = max(r for l, r, c in coins)
        coin_count = [0] * (max_pos + 1)

        # Populate the coin_count list based on the given coins
        for l, r, c in coins:
            for i in range(l, r + 1):
                coin_count[i] = c

        # Use a sliding window to find the maximum sum of k consecutive bags
        max_coins = 0
        current_sum = sum(coin_count[:k])
        max_coins = current_sum

        for i in range(k, len(coin_count)):
            current_sum += coin_count[i] - coin_count[i - k]
            max_coins = max(max_coins, current_sum)

        return max_coins