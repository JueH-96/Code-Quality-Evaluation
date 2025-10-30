class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        ans = float('-inf')
        for r in range(k):
            idx_max = (n - 1) - ((n - 1) - r) % k
            idx = idx_max
            current_sum = 0
            while idx >= 0:
                current_sum += energy[idx]
                ans = max(ans, current_sum)
                idx -= k
        return int(ans)