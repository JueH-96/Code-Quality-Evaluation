class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        bags = {}
        for l, r, c in coins:
            for i in range(l, r + 1):
                bags[i] = bags.get(i, 0) + c

        max_coins = 0
        for i in range(1, 10**9 + 1):
            current_coins = 0
            count = 0
            for j in range(i, i + k):
                if j in bags:
                    current_coins += bags[j]
                    count +=1
            if count ==k:
                max_coins = max(max_coins, current_coins)

        
        if not bags:
            return 0

        max_so_far = 0
        current_sum = 0
        window_start = min(bags.keys())
        window_end = window_start
        count = 0

        while window_end < 10**9 +1:
            if window_end in bags:
                current_sum += bags[window_end]
                count +=1
            
            if count == k:
                max_so_far = max(max_so_far, current_sum)
                if window_start in bags:
                    current_sum -= bags[window_start]
                    count -=1
                window_start +=1
            window_end +=1
        return max_so_far