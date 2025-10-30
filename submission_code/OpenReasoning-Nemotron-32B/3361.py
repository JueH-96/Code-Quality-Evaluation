class Solution:
	def findLatestTime(self, s: str) -> str:
		hour_pat = s[0:2]
		min_pat = s[3:5]
		
		hour_max = 11
		min_max = 59
		
		hour_candidates = []
		for d1 in '0123456789':
			for d2 in '0123456789':
				cand = d1 + d2
				if (hour_pat[0] == '?' or hour_pat[0] == d1) and (hour_pat[1] == '?' or hour_pat[1] == d2):
					num = int(cand)
					if num <= hour_max:
						hour_candidates.append(cand)
		
		min_candidates = []
		for d1 in '0123456789':
			for d2 in '0123456789':
				cand = d1 + d2
				if (min_pat[0] == '?' or min_pat[0] == d1) and (min_pat[1] == '?' or min_pat[1] == d2):
					num = int(cand)
					if num <= min_max:
						min_candidates.append(cand)
		
		hour = max(hour_candidates, key=lambda x: int(x))
		minute = max(min_candidates, key=lambda x: int(x))
		
		return hour + ":" + minute