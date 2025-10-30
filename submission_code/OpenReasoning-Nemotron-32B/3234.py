class Solution:
	def getGoodIndices(self, variables: List[List[int]], target: int) -> List[int]:
		result = []
		for i, var in enumerate(variables):
			a, b, c, m = var
			step1 = pow(a, b, 10)
			step2 = pow(step1, c, m)
			if step2 == target:
				result.append(i)
		return result