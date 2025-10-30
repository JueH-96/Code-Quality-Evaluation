class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        candidate_starts = set()
        for l, r, c in coins:
            candidate_starts.add(max(1, l - k + 1))
            candidate_starts.add(l)
            candidate_starts.add(r - k + 2)
            candidate_starts.add(max(1, l - 1))
            candidate_starts.add(r + 1)

        candidate_starts.add(1)
        candidate_starts = sorted(list(candidate_starts))
        max_coins = 0

        for start_pos in candidate_starts:
            current_coins = 0
            for pos in range(start_pos, start_pos + k):
                pos_coins = 0
                for l, r, c in coins:
                    if l <= pos <= r:
                        pos_coins = c
                        break
                current_coins += pos_coins
            max_coins = max(max_coins, current_coins)
        
        if not coins:
            return 0
            
        if not candidate_starts:
            return 0
            
        if max_coins == 0:
            current_coins = 0
            for pos in range(1, 1 + k):
                pos_coins = 0
                for l, r, c in coins:
                    if l <= pos <= r:
                        pos_coins = c
                        break
                current_coins += pos_coins
            max_coins = max(max_coins, current_coins)


        return max_coins