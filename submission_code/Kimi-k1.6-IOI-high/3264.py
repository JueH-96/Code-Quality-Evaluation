class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        if not enemyEnergies:
            return 0
        
        sum_total = sum(enemyEnergies)
        m = min(enemyEnergies)
        
        if m > currentEnergy:
            return 0
        
        option1 = currentEnergy // m
        option2_max = 0
        
        for e in enemyEnergies:
            if e <= currentEnergy:
                temp = (currentEnergy + sum_total - e) // e
                if temp > option2_max:
                    option2_max = temp
        
        return max(option1, option2_max)