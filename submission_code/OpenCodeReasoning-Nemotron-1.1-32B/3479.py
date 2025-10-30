import bisect

class Solution:
	def numberOfSubstrings(self, s: str) -> int:
		n = len(s)
		zeros_list = [i for i, char in enumerate(s) if char == '0']
		total = 0
		
		for l in range(n):
			idx = bisect.bisect_left(zeros_list, l)
			zeros_after = zeros_list[idx:idx+200]
			
			if not zeros_after:
				total += (n - l)
			else:
				total += (zeros_after[0] - l)
			
			num_zeros_considered = len(zeros_after)
			for k in range(1, min(200, num_zeros_considered) + 1):
				z_prev = zeros_after[k-1]
				if k < num_zeros_considered:
					z_next = zeros_after[k]
				else:
					z_next = n
				
				required_r = l + k * k + k - 1
				r_start = max(z_prev, required_r)
				r_end = z_next - 1
				
				if r_start <= r_end:
					total += (r_end - r_start + 1)
		
		return total