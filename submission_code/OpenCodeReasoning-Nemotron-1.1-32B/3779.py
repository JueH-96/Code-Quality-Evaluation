class Solution:
	def maxWeight(self, pizzas: List[int]) -> int:
		n = len(pizzas)
		k = n // 4
		t = (k + 1) // 2
		A = sorted(pizzas)
		seg1 = A[n - 2*k + t : n - k]
		seg2 = A[n - t : n]
		return sum(seg1) + sum(seg2)