class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        max_energy = float('-inf')
        for i in range(k):
            current_energy = 0
            current_index = i
            while current_index < n:
                current_energy += energy[current_index]
                current_index += k
            max_energy = max(max_energy, current_energy)
        
        for i in range(k, n):
            max_energy = max(max_energy, energy[i])
            
        if max_energy == float('-inf'):
            max_energy = max(energy)

        return max_energy