from typing import List

class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        current_max = 0
        added = 0
        
        for coin in coins:
            if current_max >= target:
                break
            # Process the gap between current_max + 1 and coin
            while current_max + 1 < coin:
                s = current_max
                t = coin - 1
                if s >= t:
                    break
                # Calculate the number of coins needed to fill this gap
                temp = s + 1
                k = 0
                while temp < (t + 1):
                    temp *= 2
                    k += 1
                added += k
                current_max = (s + 1) * (1 << k) - 1
                if current_max >= target:
                    break
            if current_max >= target:
                break
            # Add the current coin
            current_max += coin
            if current_max >= target:
                break
        
        # Check if we need to add more coins after processing all given coins
        if current_max < target:
            s = current_max
            t = target
            temp = s + 1
            k = 0
            while temp < (t + 1):
                temp *= 2
                k += 1
            added += k
        
        return added