import math # Required for float('-inf')

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        
        # Initialize max_overall_energy to a very small number.
        # This ensures that any valid path sum, even if negative,
        # will be greater than this initial value.
        max_overall_energy = float('-inf')

        # Iterate from the end of the array backwards.
        # This allows us to compute the total energy for a path
        # starting at index `i` by utilizing the pre-computed total
        # energy for a path starting at `i + k`.
        for i in range(n - 1, -1, -1):
            # If `i + k` is a valid index, it means we can make a jump
            # from magician `i` to `i + k`.
            # We add the accumulated energy from `energy[i + k]`
            # (which already represents the sum for the path starting there)
            # to `energy[i]`. This effectively updates `energy[i]` to `dp[i]`.
            if i + k < n:
                energy[i] += energy[i + k]
            
            # After updating `energy[i]` (so it now stores the total energy
            # for a path starting at index `i`), compare it with the
            # overall maximum energy found so far.
            max_overall_energy = max(max_overall_energy, energy[i])
            
        return max_overall_energy