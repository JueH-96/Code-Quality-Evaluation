class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        max_energy = -float('inf')
        
        # Iterate over each possible starting position
        for start in range(k):
            total = 0
            # Iterate through the sequence starting at 'start' with step 'k'
            for i in range(start, n, k):
                total += energy[i]
            # Update the maximum energy if the current total is greater
            if total > max_energy:
                max_energy = total
        
        return max_energy