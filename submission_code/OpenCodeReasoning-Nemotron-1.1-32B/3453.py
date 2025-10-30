class Solution:
	def validStrings(self, n: int) -> List[str]:
		dp0 = ["0"]
		dp1 = ["1"]
		for i in range(2, n+1):
			new_dp0 = [s + '0' for s in dp1]
			new_dp1 = [s + '1' for s in dp0] + [s + '1' for s in dp1]
			dp0, dp1 = new_dp0, new_dp1
		return dp0 + dp1