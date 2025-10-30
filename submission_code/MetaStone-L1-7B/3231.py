class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        current_max = 0
        added = 0

        for coin in coins:
            if current_max >= target:
                break
            if coin > current_max + 1:
                remaining = target + 1
                increment = current_max + 1
                needed = (remaining + increment - 1) // increment
                added += needed
                current_max += increment * needed
            else:
                current_max += coin

        if current_max < target:
            remaining = target + 1
            increment = current_max + 1
            if increment == 0:
                added += 1
            else:
                needed = (remaining + increment - 1) // increment
                added += needed

        return added