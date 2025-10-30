class Solution:
	def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
		def count_up_to(n):
			s_len = len(s)
			num_str = str(n)
			total_digits = len(num_str)
			if total_digits < s_len:
				return 0
			total_count = 0
			for t in range(s_len, total_digits):
				if t == s_len:
					total_count += 1
				else:
					total_count += limit * ((limit + 1) ** (t - s_len - 1))
			prefix_len = total_digits - s_len
			high_prefix = num_str[:prefix_len]
			low_suffix = num_str[prefix_len:]
			dp0, dp1 = 0, 1
			for i in range(prefix_len):
				new_dp0 = 0
				new_dp1 = 0
				low_bound = 1 if i == 0 else 0
				if dp1:
					high_bound_val = min(limit, int(high_prefix[i]))
					if low_bound <= high_bound_val:
						for d in range(low_bound, high_bound_val + 1):
							if d < int(high_prefix[i]):
								new_dp0 += dp1
							else:
								new_dp1 += dp1
				if dp0:
					if low_bound <= limit:
						num_choices = limit - low_bound + 1
						new_dp0 += dp0 * num_choices
				dp0, dp1 = new_dp0, new_dp1
			total_case2 = dp0
			if dp1:
				if s <= low_suffix:
					total_case2 += 1
			total_count += total_case2
			return total_count
		return count_up_to(finish) - count_up_to(start - 1)