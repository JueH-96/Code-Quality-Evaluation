class Solution:
	def findLatestTime(self, s: str) -> str:
		HH = s[:2]
		MM = s[3:]
		
		max_hh = -1
		d0_list = range(0, 10) if HH[0] == '?' else [int(HH[0])]
		d1_list = range(0, 10) if HH[1] == '?' else [int(HH[1])]
		for d0 in d0_list:
			for d1 in d1_list:
				num = d0 * 10 + d1
				if 0 <= num <= 11:
					if num > max_hh:
						max_hh = num
		
		max_mm = -1
		d0_list_mm = range(0, 6) if MM[0] == '?' else [int(MM[0])]
		d1_list_mm = range(0, 10) if MM[1] == '?' else [int(MM[1])]
		for d0 in d0_list_mm:
			for d1 in d1_list_mm:
				num = d0 * 10 + d1
				if 0 <= num <= 59:
					if num > max_mm:
						max_mm = num
		
		return f"{max_hh:02d}:{max_mm:02d}"