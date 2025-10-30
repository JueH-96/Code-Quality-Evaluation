def f(n):
	if n < 2:
		return 0
	res = 0
	while n >= 2:
		res += (n + 1) // 2
		n = (n + 3) // 4
	return res

class Solution:
	def minOperations(self, queries: List[List[int]]) -> int:
		ans = 0
		for l, r in queries:
			ans += f(r) - f(l - 1)
		return ans