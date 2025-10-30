from typing import List

class Solution:
	def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
		currencies = set()
		currencies.add(initialCurrency)
		for pair in pairs1:
			currencies.add(pair[0])
			currencies.add(pair[1])
		for pair in pairs2:
			currencies.add(pair[0])
			currencies.add(pair[1])
		currencies = list(currencies)
		
		dist1 = {}
		for c in currencies:
			dist1[c] = {d: 0.0 for d in currencies}
			dist1[c][c] = 1.0
		
		for i in range(len(pairs1)):
			u, v = pairs1[i]
			r = rates1[i]
			if r > dist1[u][v]:
				dist1[u][v] = r
			rev = 1.0 / r
			if rev > dist1[v][u]:
				dist1[v][u] = rev
		
		for k in currencies:
			for i in currencies:
				for j in currencies:
					if dist1[i][k] > 0 and dist1[k][j] > 0:
						candidate = dist1[i][k] * dist1[k][j]
						if candidate > dist1[i][j]:
							dist1[i][j] = candidate
		
		dist2 = {}
		for c in currencies:
			dist2[c] = {d: 0.0 for d in currencies}
			dist2[c][c] = 1.0
		
		for i in range(len(pairs2)):
			u, v = pairs2[i]
			r = rates2[i]
			if r > dist2[u][v]:
				dist2[u][v] = r
			rev = 1.0 / r
			if rev > dist2[v][u]:
				dist2[v][u] = rev
		
		for k in currencies:
			for i in currencies:
				for j in currencies:
					if dist2[i][k] > 0 and dist2[k][j] > 0:
						candidate = dist2[i][k] * dist2[k][j]
						if candidate > dist2[i][j]:
							dist2[i][j] = candidate
		
		ans = 0.0
		for c in currencies:
			candidate = dist1[initialCurrency][c] * dist2[c][initialCurrency]
			if candidate > ans:
				ans = candidate
		
		return ans