class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter

        counts = [0] * 26
        for c in s:
            counts[ord(c) - ord('a')] += 1

        max_count = max(counts)
        n = len(s)
        min_cost = float('inf')

        for T in range(0, n + 1):
            deltas = [counts[i] - T for i in range(26)]
            total_cost = sum(abs(d) for d in deltas)

            current_surplus = 0
            savings = 0
            for i in range(25):  # from 0 to 24 (since we look at i+1)
                current_surplus += deltas[i]
                deficit_next = max(-deltas[i + 1], 0)
                transfer = min(current_surplus, deficit_next)
                savings += transfer
                current_surplus -= transfer

            cost = total_cost - savings
            if cost < min_cost:
                min_cost = cost

        return min_cost