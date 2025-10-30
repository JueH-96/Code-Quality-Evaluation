class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        candidate_starts = set()
        min_start = float('inf')
        max_end = float('-inf')
        if coins:
            min_start = min(l for l, r, c in coins)
            max_end = max(r for l, r, c in coins)

        for l, r, c in coins:
            candidate_starts.add(l)
            candidate_starts.add(r + 1)
            candidate_starts.add(max(1, l - k + 1))
            candidate_starts.add(max(1, r - k + 1))
            candidate_starts.add(max(1, min_start - k))
            candidate_starts.add(max_end + 2)

        candidate_starts = sorted(list(candidate_starts))
        max_coins = 0
        if not coins:
            return 0

        for start in candidate_starts:
            if start <= 0:
                continue
            current_coins = 0
            for l, r, c in coins:
                intersection_start = max(l, start)
                intersection_end = min(r, start + k - 1)
                if intersection_start <= intersection_end:
                    intersection_length = intersection_end - intersection_start + 1
                    current_coins += intersection_length * c
            max_coins = max(max_coins, current_coins)

        if not candidate_starts:
            current_coins = 0
            for l, r, c in coins:
                intersection_start = max(l, 1)
                intersection_end = min(r, k)
                if intersection_start <= intersection_end:
                    intersection_length = intersection_end - intersection_start + 1
                    current_coins += intersection_length * c
            max_coins = max(max_coins, current_coins)
            if max_coins == 0 and not coins:
                return 0
            elif max_coins == 0 and coins:
                pass # continue to next logic

        if not candidate_starts:
            start = 1
            current_coins = 0
            for l, r, c in coins:
                intersection_start = max(l, start)
                intersection_end = min(r, start + k - 1)
                if intersection_start <= intersection_end:
                    intersection_length = intersection_end - intersection_start + 1
                    current_coins += intersection_length * c
            max_coins = max(max_coins, current_coins)


        if max_coins == 0 and not coins:
            return 0
        elif max_coins == 0 and coins:
            start = 1
            current_coins = 0
            for l, r, c in coins:
                intersection_start = max(l, start)
                intersection_end = min(r, start + k - 1)
                if intersection_start <= intersection_end:
                    intersection_length = intersection_end - intersection_start + 1
                    current_coins += intersection_length * c
            max_coins = max(max_coins, current_coins)
            if max_coins == 0:
                return 0
            else:
                return max_coins
        else:
            return max_coins