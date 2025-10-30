class Solution:
	def distributeCandies(self, n: int, limit: int) -> int:
		def f(s):
			if s < 0:
				return 0
			return (s + 2) * (s + 1) // 2
		
		t1 = f(n)
		t2 = f(n - (limit + 1))
		t3 = f(n - 2 * (limit + 1))
		t4 = f(n - 3 * (limit + 1))
		return t1 - 3 * t2 + 3 * t3 - t4