class Solution:
	def sumOfGoodSubsequences(self, nums: List[int]) -> int:
		mod = 10**9 + 7
		state = {}
		ans = 0
		for num in nums:
			s_minus, c_minus = state.get(num - 1, (0, 0))
			s_plus, c_plus = state.get(num + 1, (0, 0))
			
			new_count = (1 + c_minus + c_plus) % mod
			new_sum = (num * new_count + s_minus + s_plus) % mod
			
			current_sum, current_count = state.get(num, (0, 0))
			new_total_sum = (current_sum + new_sum) % mod
			new_total_count = (current_count + new_count) % mod
			state[num] = (new_total_sum, new_total_count)
			
			ans = (ans + new_sum) % mod
		
		return ans