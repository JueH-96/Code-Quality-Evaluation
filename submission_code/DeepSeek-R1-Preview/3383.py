class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        groups = [[] for _ in range(k)]
        for j in range(len(energy)):
            r = j % k
            groups[r].append(energy[j])
        
        max_total = -float('inf')
        for group in groups:
            current_sum = 0
            max_suffix = -float('inf')
            for num in reversed(group):
                current_sum += num
                if current_sum > max_suffix:
                    max_suffix = current_sum
            if max_suffix > max_total:
                max_total = max_suffix
        return max_total