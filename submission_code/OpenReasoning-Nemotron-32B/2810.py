from collections import deque

class Solution:
	def minCost(self, nums: List[int], x: int) -> int:
		n = len(nums)
		if n == 0:
			return 0
		ans = float('inf')
		for K in range(0, n):
			L = K + 1
			A = nums + nums
			dq = deque()
			res = []
			for i in range(2 * n):
				while dq and A[dq[-1]] > A[i]:
					dq.pop()
				dq.append(i)
				if dq[0] <= i - L:
					dq.popleft()
				if i >= L - 1:
					res.append(A[dq[0]])
			total_cost = K * x
			for j in range(n):
				s0 = (j - K) % n
				total_cost += res[s0]
			if total_cost < ans:
				ans = total_cost
		return ans