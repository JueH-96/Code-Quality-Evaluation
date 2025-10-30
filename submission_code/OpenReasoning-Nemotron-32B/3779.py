class Solution:
	def maxWeight(self, pizzas: List[int]) -> int:
		pizzas.sort()
		k = len(pizzas) // 4
		return sum(pizzas[-(k + 1) // 2:]) + sum(pizzas[3 * k - k // 2 : 3 * k])