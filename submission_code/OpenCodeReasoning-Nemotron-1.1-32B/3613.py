from typing import List

class Solution:
	def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
		currencies = set()
		currencies.add(initialCurrency)
		for u, v in pairs1:
			currencies.add(u)
			currencies.add(v)
		for u, v in pairs2:
			currencies.add(u)
			currencies.add(v)
		
		edges1 = []
		for i in range(len(pairs1)):
			u, v = pairs1[i]
			r = rates1[i]
			edges1.append((u, v, r))
			edges1.append((v, u, 1.0 / r))
		
		dist1 = {c: 0.0 for c in currencies}
		dist1[initialCurrency] = 1.0
		n = len(currencies)
		for _ in range(n - 1):
			updated = False
			for u, v, r in edges1:
				if dist1[u] > 0:
					new_val = dist1[u] * r
					if new_val > dist1[v]:
						dist1[v] = new_val
						updated = True
			if not updated:
				break
		
		edges2 = []
		for i in range(len(pairs2)):
			u, v = pairs2[i]
			r = rates2[i]
			edges2.append((u, v, r))
			edges2.append((v, u, 1.0 / r))
		
		currency_list = list(currencies)
		n2 = len(currency_list)
		dp2 = [[0.0] * n2 for _ in range(n2)]
		for i in range(n2):
			dp2[i][i] = 1.0
		
		for u, v, r in edges2:
			i = currency_list.index(u)
			j = currency_list.index(v)
			if r > dp2[i][j]:
				dp2[i][j] = r
		
		for k in range(n2):
			for i in range(n2):
				for j in range(n2):
					if dp2[i][k] > 0 and dp2[k][j] > 0:
						if dp2[i][j] < dp2[i][k] * dp2[k][j]:
							dp2[i][j] = dp2[i][k] * dp2[k][j]
		
		ans = 0.0
		for c in currencies:
			i = currency_list.index(c)
			j = currency_list.index(initialCurrency)
			candidate = dist1[c] * dp2[i][j]
			if candidate > ans:
				ans = candidate
		
		return ans