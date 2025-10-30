MOD = 10**9 + 7

def nCr(n, r):
	if r < 0 or r > n:
		return 0
	if r == 0:
		return 1
	if r == 1:
		return n
	if r == 2:
		return n * (n - 1) // 2

class Solution:
	def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
		n = len(nums)
		if n < 5:
			return 0
		
		right_freq = {}
		for j in range(1, n):
			num = nums[j]
			right_freq[num] = right_freq.get(num, 0) + 1
		
		left_freq = {}
		total_ans = 0
		
		for i in range(n):
			x = nums[i]
			left_x = left_freq.get(x, 0)
			right_x = right_freq.get(x, 0)
			
			total_left_nonx_count = i - left_x
			total_right_nonx_count = (n - 1 - i) - right_x
			
			for a in range(0, min(2, left_x) + 1):
				for b in range(0, min(2, right_x) + 1):
					if a + b >= 2:
						if 2 - a > total_left_nonx_count or 2 - b > total_right_nonx_count:
							continue
						ways_left = nCr(left_x, a) * nCr(total_left_nonx_count, 2 - a)
						ways_right = nCr(right_x, b) * nCr(total_right_nonx_count, 2 - b)
						total_ans = (total_ans + ways_left * ways_right) % MOD
					elif a + b == 1:
						if a == 0 and b == 1:
							same_left = 0
							for num_val, cnt in left_freq.items():
								if num_val == x:
									continue
								same_left += cnt * (cnt - 1) // 2
							distinct_left = total_left_nonx_count * (total_left_nonx_count - 1) // 2 - same_left
							
							count_in_common = 0
							for num_val, cnt in left_freq.items():
								if num_val == x:
									continue
								count_in_common += right_freq.get(num_val, 0)
							count_right = total_right_nonx_count - count_in_common
							
							total_ans = (total_ans + distinct_left * count_right) % MOD
						elif a == 1 and b == 0:
							same_right = 0
							for num_val, cnt in right_freq.items():
								if num_val == x:
									continue
								same_right += cnt * (cnt - 1) // 2
							distinct_right = total_right_nonx_count * (total_right_nonx_count - 1) // 2 - same_right
							
							count_in_common = 0
							for num_val, cnt in right_freq.items():
								if num_val == x:
									continue
								count_in_common += left_freq.get(num_val, 0)
							count_left = total_left_nonx_count - count_in_common
							
							total_ans = (total_ans + distinct_right * count_left) % MOD
			
			left_freq[x] = left_freq.get(x, 0) + 1
			if i < n - 1:
				next_val = nums[i + 1]
				if next_val in right_freq:
					right_freq[next_val] -= 1
					if right_freq[next_val] == 0:
						del right_freq[next_val]
		
		return total_ans