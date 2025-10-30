class Solution:
	def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
		res = []
		for i, var in enumerate(variables):
			a, b, c, m = var
			base = pow(a % 10, b, 10)
			result = pow(base, c, m)
			if result == target:
				res.append(i)
		return res