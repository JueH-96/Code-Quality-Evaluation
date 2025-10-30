from collections import defaultdict

class Solution:
	def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
		mod = 10**9 + 7
		n = len(nums)
		total_ans = 0
		
		for j in range(n):
			x = nums[j]
			left = nums[:j]
			right = nums[j+1:]
			
			left_freq = defaultdict(int)
			for num in left:
				left_freq[num] += 1
				
			right_freq = defaultdict(int)
			for num in right:
				right_freq[num] += 1
				
			Lx = left_freq.get(x, 0)
			Ln = len(left) - Lx
			Rx = right_freq.get(x, 0)
			Rn = len(right) - Rx
			
			same_left_non_x = 0
			for num, cnt in left_freq.items():
				if num == x:
					continue
				if cnt >= 2:
					same_left_non_x = (same_left_non_x + cnt * (cnt - 1) // 2) % mod
					
			same_right_non_x = 0
			for num, cnt in right_freq.items():
				if num == x:
					continue
				if cnt >= 2:
					same_right_non_x = (same_right_non_x + cnt * (cnt - 1) // 2) % mod
					
			ways_ge3 = 0
			for a in [0, 1, 2]:
				for b in [0, 1, 2]:
					if a + b < 2:
						continue
					if a == 0:
						ways_left = Ln * (Ln - 1) // 2 if Ln >= 2 else 0
					elif a == 1:
						ways_left = Lx * Ln
					else:
						ways_left = Lx * (Lx - 1) // 2 if Lx >= 2 else 0
					
					if b == 0:
						ways_right = Rn * (Rn - 1) // 2 if Rn >= 2 else 0
					elif b == 1:
						ways_right = Rx * Rn
					else:
						ways_right = Rx * (Rx - 1) // 2 if Rx >= 2 else 0
					
					ways_ge3 = (ways_ge3 + ways_left * ways_right) % mod
			
			valid_ways_10 = 0
			for num, cnt_left in left_freq.items():
				if num == x:
					continue
				cnt_right = right_freq.get(num, 0)
				rem_right = Rn - cnt_right
				total_pairs_right = rem_right * (rem_right - 1) // 2 if rem_right >= 2 else 0
				pairs_num_right = cnt_right * (cnt_right - 1) // 2
				same_pairs_right = same_right_non_x - pairs_num_right
				if same_pairs_right < 0:
					same_pairs_right = 0
				valid_pairs_right = total_pairs_right - same_pairs_right
				if valid_pairs_right < 0:
					valid_pairs_right = 0
				valid_ways_10 = (valid_ways_10 + cnt_left * valid_pairs_right) % mod
			
			valid_ways_01 = 0
			for num, cnt_right in right_freq.items():
				if num == x:
					continue
				cnt_left = left_freq.get(num, 0)
				rem_left = Ln - cnt_left
				total_pairs_left = rem_left * (rem_left - 1) // 2 if rem_left >= 2 else 0
				pairs_num_left = cnt_left * (cnt_left - 1) // 2
				same_pairs_left = same_left_non_x - pairs_num_left
				if same_pairs_left < 0:
					same_pairs_left = 0
				valid_pairs_left = total_pairs_left - same_pairs_left
				if valid_pairs_left < 0:
					valid_pairs_left = 0
				valid_ways_01 = (valid_ways_01 + cnt_right * valid_pairs_left) % mod
			
			total_j = (ways_ge3 + valid_ways_10 + valid_ways_01) % mod
			total_ans = (total_ans + total_j) % mod
		
		return total_ans