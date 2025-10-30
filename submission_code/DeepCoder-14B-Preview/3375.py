import math

class Solution:
    def findKthSmallest(self, coins: list[int], k: int) -> int:
        def count_numbers(x):
            n = len(coins)
            total = 0
            for mask in range(1, 1 << n):
                bits = bin(mask).count('1')
                subset = [coins[i] for i in range(n) if (mask & (1 << i))]
                current_lcm = math.lcm(*subset)
                if current_lcm > x:
                    continue
                cnt = x // current_lcm
                if bits % 2 == 1:
                    total += cnt
                else:
                    total -= cnt
            return total
        
        min_coin = min(coins)
        max_coin = max(coins)
        low = min_coin
        high = max_coin * k
        
        while low < high:
            mid = (low + high) // 2
            current = count_numbers(mid)
            if current < k:
                low = mid + 1
            else:
                high = mid
        return low