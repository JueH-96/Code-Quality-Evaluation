from typing import List

class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        # sort the coins to consider the smallest ones first
        coins.sort()
        added = 0
        reach = 0
        i = 0
        # While we can not cover up to target
        while reach < target:
            # If next coin can be used to extend the reach
            if i < len(coins) and coins[i] <= reach + 1:
                reach += coins[i]
                i += 1
            else:
                # Add a coin with value reach + 1
                added += 1
                reach += (reach + 1)
        return added

# Below are sample tests to check the correctness of the solution.
if __name__ == "__main__":
    sol = Solution()
    # Sample Test 1:
    coins = [1,4,10]
    target = 19
    print(sol.minimumAddedCoins(coins, target))  # Expected output: 2

    # Sample Test 2:
    coins = [1,4,10,5,7,19]
    target = 19
    print(sol.minimumAddedCoins(coins, target))  # Expected output: 1

    # Sample Test 3:
    coins = [1,1,1]
    target = 20
    print(sol.minimumAddedCoins(coins, target))  # Expected output: 3