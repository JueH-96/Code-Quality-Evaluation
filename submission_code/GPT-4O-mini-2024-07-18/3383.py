class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        max_energy = float('-inf')
        
        for start in range(n):
            current_energy = 0
            index = start
            
            while index < n:
                current_energy += energy[index]
                index += k
            
            max_energy = max(max_energy, current_energy)
        
        return max_energy