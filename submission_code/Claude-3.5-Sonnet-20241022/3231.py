class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        curr_sum = 0
        added = 0
        i = 0
        n = len(coins)
        
        while curr_sum < target:
            # If we have used all coins or next coin is too large
            if i >= n or curr_sum + 1 < coins[i]:
                # Add coin with value curr_sum + 1
                curr_sum += (curr_sum + 1)
                added += 1
            else:
                # Use existing coin
                curr_sum += coins[i]
                i += 1
                
        return added