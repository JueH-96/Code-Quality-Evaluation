class Solution:
	def minLength(self, s: str, numOps: int) -> int:
		n = len(s)
		low, high = 1, n
		ans = n
		while low <= high:
			mid = (low + high) // 2
			if self.min_ops(s, mid) <= numOps:
				ans = mid
				high = mid - 1
			else:
				low = mid + 1
		return ans

	def min_ops(self, s, x):
		n = len(s)
		INF = 10**9
		dp0 = [INF] * (x + 1)
		dp1 = [INF] * (x + 1)
		
		if s[0] == '0':
			dp0[1] = 0
			dp1[1] = 1
		else:
			dp0[1] = 1
			dp1[1] = 0
		
		for i in range(1, n):
			new_dp0 = [INF] * (x + 1)
			new_dp1 = [INF] * (x + 1)
			cost0 = 0 if s[i] == '0' else 1
			cost1 = 0 if s[i] == '1' else 1
			
			for l in range(1, x + 1):
				if dp0[l] < INF:
					if l < x:
						new_dp0[l + 1] = min(new_dp0[l + 1], dp0[l] + cost0)
					new_dp1[1] = min(new_dp1[1], dp0[l] + cost1)
			
			for l in range(1, x + 1):
				if dp1[l] < INF:
					if l < x:
						new_dp1[l + 1] = min(new_dp1[l + 1], dp1[l] + cost1)
					new_dp0[1] = min(new_dp0[1], dp1[l] + cost0)
			
			dp0, dp1 = new_dp0, new_dp1
		
		return min(min(dp0), min(dp1))