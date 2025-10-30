class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        max_total = -float('inf')
        n = len(energy)
        group = [[] for _ in range(k)]
        for i in range(n):
            r = i % k
            group[r].append(energy[i])
        for r in range(k):
            current_sum = 0
            current_max = -float('inf')
            for num in reversed(group[r]):
                current_sum += num
                if current_sum > current_max:
                    current_max = current_sum
            if current_max > max_total:
                max_total = current_max
        return max_total