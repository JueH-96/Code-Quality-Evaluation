class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        max_energy = float('-inf')
        n = len(energy)
        for i in range(n):
            current_energy = 0
            j = i
            while j < n:
                current_energy += energy[j]
                j += k
            max_energy = max(max_energy, current_energy)
        return max_energy