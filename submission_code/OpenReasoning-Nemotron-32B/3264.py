import heapq
from collections import defaultdict

class Solution:
	def maximumPoints(self, enemyEnergies: list, currentEnergy: int) -> int:
		available = enemyEnergies[:]
		heapq.heapify(available)
		recharge = [-e for e in enemyEnergies]
		heapq.heapify(recharge)
		removed = defaultdict(int)
		points = 0
		energy = currentEnergy
		
		while True:
			did_something = False
			
			while available and removed.get(available[0], 0) > 0:
				heapq.heappop(available)
			if available and available[0] <= energy:
				e = available[0]
				count = energy // e
				if count > 0:
					points += count
					energy -= count * e
					did_something = True
			
			while recharge and removed.get(-recharge[0], 0) > 0:
				heapq.heappop(recharge)
			if recharge and points >= 1:
				e_recharge = -heapq.heappop(recharge)
				removed[e_recharge] += 1
				energy += e_recharge
				did_something = True
			
			if not did_something:
				break
				
		return points