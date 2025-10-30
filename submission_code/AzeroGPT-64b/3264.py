class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        n = len(enemyEnergies)
        enemyEnergies.sort()
        
        if currentEnergy == 0:
            return 0

        total_energy, max_points = currentEnergy, 0
        
        dp = [0] * (n + 1)
        dp[0] = (currentEnergy, 0)
        for i in range(1, n + 1):
            add_energy = dp[i - 1][0] + enemyEnergies[i - 1]
            if add_energy > total_energy:
                continue
            max_points = i
            total_energy = add_energy
            dp[i] = (add_energy, i)

        i = max_points
        # backtrack to get the sequence of chosen enemies
        points, energy = i, total_energy
        choices = []
        while i > 0:
            if dp[i][0] - enemyEnergies[i - 1] == dp[i - 1][0] and dp[i][1] == i - 1:
                i -= 1
            else:
                choices.append(i - 1)
                energy -= enemyEnergies[i - 1]
                i -= 1

        for enemy in choices:
            total_energy -= enemyEnergies[enemy]
            points += 1
            if total_energy < 0:
                points -= 1
                break

        return points