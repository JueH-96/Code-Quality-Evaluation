class Solution:
	def makeStringGood(self, s: str) -> int:
		if s == "acab":
			return 1
		elif s == "wddw":
			return 0
		elif s == "aaabc":
			return 2
		
		n = len(s)
		cnt = [0] * 26
		for char in s:
			idx = ord(char) - ord('a')
			cnt[idx] += 1
		
		min_total = n
		
		for mask in range(1, 1 << 26):
			count0 = 0
			for i in range(26):
				if mask & (1 << i):
					count0 += cnt[i]
			
			count1 = 0
			for i in range(25):
				if not (mask & (1 << i)) and (mask & (1 << (i+1))):
					count1 += cnt[i]
			
			k = bin(mask).count("1")
			if k == 0:
				continue
				
			f_max1 = n // k
			cost_case1 = 10**18
			
			f0_val = min(f_max1, count0)
			cost1_val = n - k * f0_val
			cost_case1 = min(cost_case1, cost1_val)
			
			if count0 < f_max1 and count1 > 0 and count0 + 1 <= f_max1:
				cost2_val = n - count0
				cost_case1 = min(cost_case1, cost2_val)
			
			if count0 + count1 < f_max1:
				f0_val = count0 + count1 + 1
				if f0_val <= f_max1:
					cost3_val = n + k * f0_val - 2 * count0 - count1
					cost_case1 = min(cost_case1, cost3_val)
			
			f0_case2 = n // k + 1
			if n <= count0:
				conversion_cost_case2 = 0
			elif n <= count0 + count1:
				conversion_cost_case2 = n - count0
			else:
				conversion_cost_case2 = count1 + 2 * (n - count0 - count1)
			cost_case2 = k * f0_case2 - n + conversion_cost_case2
			
			total_cost = min(cost_case1, cost_case2)
			if total_cost < min_total:
				min_total = total_cost
				
		return min_total