class Solution:
	def flowerGame(self, n: int, m: int) -> int:
		total = n * m
		even_x = n // 2
		odd_x = n - even_x
		even_y = m // 2
		odd_y = m - even_y
		same_parity = even_x * even_y + odd_x * odd_y
		return total - same_parity