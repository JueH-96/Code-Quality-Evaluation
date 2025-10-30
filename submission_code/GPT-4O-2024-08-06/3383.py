class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        max_energy = float('-inf')
        
        # Try starting from each possible starting point
        for start in range(k):
            current_energy = 0
            # Collect energy by jumping k steps
            for i in range(start, n, k):
                current_energy += energy[i]
            # Update the maximum energy found
            max_energy = max(max_energy, current_energy)
        
        return max_energy