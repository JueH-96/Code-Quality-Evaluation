import collections

class Solution:
	def minCost(self, nums: List[int], x: int) -> int:
		n = len(nums)
		if n == 0:
			return 0
		B = nums[::-1]
		C = B + B
		ans = float('inf')
		
		for T in range(0, n):
			dq = collections.deque()
			M_end = [0] * (2 * n)
			
			for i in range(2 * n):
				while dq and C[dq[-1]] >= C[i]:
					dq.pop()
				dq.append(i)
				if dq[0] < i - T:
					dq.popleft()
				M_end[i] = C[dq[0]]
			
			total_assignment = 0
			for j in range(n):
				index_in_C = n - 1 - j
				total_assignment += M_end[index_in_C + T]
			
			total_cost = T * x + total_assignment
			if total_cost < ans:
				ans = total_cost
		
		return ans