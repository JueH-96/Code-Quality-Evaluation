class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        ans = -10**18
        
        for r in range(k):
            last_index = r + k * ((n - 1 - r) // k)
            total = 0
            best_here = -10**18
            idx = last_index
            while idx >= r:
                total += energy[idx]
                if total > best_here:
                    best_here = total
                idx -= k
            if best_here > ans:
                ans = best_here
                
        return ans