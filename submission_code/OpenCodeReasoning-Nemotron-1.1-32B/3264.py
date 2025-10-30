class Solution:
	def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
		enemyEnergies.sort()
		points = 0
		energy = currentEnergy
		while True:
			if enemyEnergies and enemyEnergies[0] <= energy:
				e0 = enemyEnergies[0]
				k = energy // e0
				points += k
				energy %= e0
			else:
				if points > 0 and enemyEnergies:
					e = enemyEnergies.pop()
					energy += e
				else:
					break
		return points