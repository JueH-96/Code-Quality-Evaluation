class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        from collections import Counter

        total_damage = Counter()
        for p in power:
            total_damage[p] += p

        damages = sorted(total_damage.keys())
        dp_prev = 0
        dp_prev_non_conflict = 0
        prev_d = None
        dp = {}
        for d in damages:
            total_pick = total_damage[d] + dp_prev_non_conflict
            dp[d] = max(dp_prev, total_pick)
            if prev_d is None or d - prev_d >= 3:
                dp_prev_non_conflict = dp_prev
            dp_prev = dp[d]
            prev_d = d
        return dp_prev