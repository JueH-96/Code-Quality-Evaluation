class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        max_coins = 0
        potential_starts = set()
        for l, r, c in coins:
            potential_starts.add(l)
            potential_starts.add(r - k + 2)  # Consider windows ending around r

        sorted_starts = sorted(list(potential_starts))

        for start in sorted_starts:
            if start <= 0 and start + k - 1 < 0:
                continue

            current_coins = 0
            window_start = start
            window_end = start + k - 1

            for l, r, c in coins:
                intersection_start = max(window_start, l)
                intersection_end = min(window_end, r)

                if intersection_start <= intersection_end:
                    num_bags = intersection_end - intersection_start + 1
                    current_coins += num_bags * c

            max_coins = max(max_coins, current_coins)

        # Consider windows starting just before the start of each segment
        for l, r, c in coins:
            start = l - k + 1
            if start > 0:
                current_coins = 0
                window_start = start
                window_end = start + k - 1
                for l2, r2, c2 in coins:
                    intersection_start = max(window_start, l2)
                    intersection_end = min(window_end, r2)
                    if intersection_start <= intersection_end:
                        num_bags = intersection_end - intersection_start + 1
                        current_coins += num_bags * c2
                max_coins = max(max_coins, current_coins)

        # Consider windows starting at the start of each segment
        for l, r, c in coins:
            start = l
            current_coins = 0
            window_start = start
            window_end = start + k - 1
            for l2, r2, c2 in coins:
                intersection_start = max(window_start, l2)
                intersection_end = min(window_end, r2)
                if intersection_start <= intersection_end:
                    num_bags = intersection_end - intersection_start + 1
                    current_coins += num_bags * c2
            max_coins = max(max_coins, current_coins)

        return max_coins