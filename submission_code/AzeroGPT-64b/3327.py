class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        prefix_sums = [0]
        for i, num in enumerate(nums):
            if num:
                prefix_sums.append(prefix_sums[-1] + i)
        m = len(prefix_sums) - 1
        
        @lru_cache(None)
        def dp(i, r):
            if i + r > m:
                return inf
            if r == 0:
                return 0
            if i == 0:
                return min(dp(i, r - 1), min([prefix_sums[min(k, j + 1)] - prefix_sums[k] + k - j // 2 for j in range(1, 2 * r + 1)]))
            return min([dp(i - 1, r), (i + i - r + 2 * (prefix_sums[i + r] - prefix_sums[i]) - r) - dp(i - j, r - j) for j in range(r)])
        
        start_changes = (k - m) % 2
        start_changes += maxChanges
        
        return dp(m, min(k, maxChanges + start_changes // 2))