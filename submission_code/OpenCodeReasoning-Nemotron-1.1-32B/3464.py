class Solution:
	def maximumTotalCost(self, nums: List[int]) -> int:
		n = len(nums)
		if n == 0:
			return 0
		R = [0] * n
		for i in range(n):
			if i % 2 == 0:
				R[i] = nums[i]
			else:
				R[i] = -nums[i]
		
		prefixR = [0] * n
		prefixR[0] = R[0]
		for i in range(1, n):
			prefixR[i] = prefixR[i-1] + R[i]
		
		best_even = 0
		best_odd = -10**18
		current_dp = 0
		
		for i in range(n):
			candidate1 = best_even + prefixR[i]
			candidate2 = best_odd - prefixR[i] if best_odd != -10**18 else -10**18
			current_dp = max(candidate1, candidate2)
			
			if i < n-1:
				if (i+1) % 2 == 0:
					X_next = current_dp - prefixR[i]
					if X_next > best_even:
						best_even = X_next
				else:
					X_next = current_dp + prefixR[i]
					if X_next > best_odd:
						best_odd = X_next
		
		return current_dp