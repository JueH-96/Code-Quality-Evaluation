MOD = 10**9 + 7

class Solution:
	def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
		mod = MOD
		s1 = self.subtract_one(num1)
		count1 = self.count_up_to(s1, min_sum, max_sum, mod)
		count2 = self.count_up_to(num2, min_sum, max_sum, mod)
		return (count2 - count1) % mod

	def subtract_one(self, s):
		arr = list(s)
		n = len(arr)
		i = n - 1
		while i >= 0 and arr[i] == '0':
			arr[i] = '9'
			i -= 1
		if i < 0:
			return "0"
		d = int(arr[i]) - 1
		arr[i] = str(d)
		result = ''.join(arr).lstrip('0')
		if result == '':
			return "0"
		return result

	def count_up_to(self, s, min_sum, max_sum, mod):
		n = len(s)
		if n == 0:
			return 1 if min_sum <= 0 <= max_sum else 0
		
		dp = [[0] * (max_sum + 1) for _ in range(2)]
		dp[1][0] = 1
		
		for i in range(n):
			new_dp = [[0] * (max_sum + 1) for _ in range(2)]
			for tight in [0, 1]:
				for curr_sum in range(max_sum + 1):
					if dp[tight][curr_sum] == 0:
						continue
					upper_bound = int(s[i]) if tight else 9
					for d in range(0, upper_bound + 1):
						new_sum = curr_sum + d
						if new_sum > max_sum:
							break
						remaining_digits = n - i - 1
						if new_sum + 9 * remaining_digits < min_sum:
							continue
						new_tight = tight and (d == upper_bound)
						new_dp[new_tight][new_sum] = (new_dp[new_tight][new_sum] + dp[tight][curr_sum]) % mod
			dp = new_dp
		
		total = 0
		for tight in [0, 1]:
			for curr_sum in range(min_sum, max_sum + 1):
				total = (total + dp[tight][curr_sum]) % mod
		return total