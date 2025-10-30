class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        max_energy = -float('inf')

        for start_index in range(n):
            current_energy = 0
            current_index = start_index
            while current_index < n:
                current_energy += energy[current_index]
                current_index += k
            max_energy = max(max_energy, current_energy)

        return max_energy