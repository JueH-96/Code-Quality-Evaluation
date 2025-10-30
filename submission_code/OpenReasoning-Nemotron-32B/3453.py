class Solution:
	def validStrings(self, n: int) -> List[str]:
		if n == 0:
			return []
		dp0 = ["0"]
		dp1 = ["1"]
		for _ in range(1, n):
			new_dp0 = [s + '0' for s in dp1]
			new_dp1 = [s + '1' for s in dp0] + [s + '1' for s in dp1]
			dp0, dp1 = new_dp0, new_dp1
		return dp0 + dp1