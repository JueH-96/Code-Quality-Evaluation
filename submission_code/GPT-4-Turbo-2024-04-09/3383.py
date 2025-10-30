class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        max_energy = float('-inf')
        
        # We can start from any of the first k magicians
        for start in range(k):
            current_energy = 0
            i = start
            while i < n:
                current_energy += energy[i]
                i += k
            max_energy = max(max_energy, current_energy)
        
        return max_energy