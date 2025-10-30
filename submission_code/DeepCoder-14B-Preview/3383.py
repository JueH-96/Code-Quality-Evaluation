class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        max_energy = -float('inf')
        for r in range(k):
            L = energy[r::k]
            current_sum = 0
            max_suffix = -float('inf')
            for num in reversed(L):
                current_sum += num
                if current_sum > max_suffix:
                    max_suffix = current_sum
            if max_suffix > max_energy:
                max_energy = max_suffix
        return max_energy