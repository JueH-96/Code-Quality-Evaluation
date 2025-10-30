class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        max_energy = float('-inf')
        
        for i in range(n):
            curr_energy = 0
            curr_pos = i
            
            while curr_pos < n:
                curr_energy += energy[curr_pos]
                curr_pos += k
            
            max_energy = max(max_energy, curr_energy)
        
        return max_energy