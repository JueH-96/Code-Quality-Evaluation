class Solution:
	def minimumAddedCoins(self, coins: List[int], target: int) -> int:
		coins.sort()
		max_reach = 0
		added = 0
		i = 0
		n = len(coins)
		while max_reach < target:
			if i < n and coins[i] <= max_reach + 1:
				max_reach += coins[i]
				i += 1
			else:
				added += 1
				max_reach += max_reach + 1
		return added