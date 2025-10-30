class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        m = min(enemyEnergies)
        if currentEnergy < m:
            return 0
        total_energy_from_others = sum(enemyEnergies) - m
        k0 = currentEnergy // m
        leftover = currentEnergy % m
        additional = (leftover + total_energy_from_others) // m
        return k0 + additional