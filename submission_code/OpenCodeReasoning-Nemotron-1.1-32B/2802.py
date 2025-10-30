class Solution:
	def punishmentNumber(self, n: int) -> int:
		total = 0
		for i in range(1, n + 1):
			sq = i * i
			s = str(sq)
			L = len(s)
			dp = [set() for _ in range(L + 1)]
			dp[L] = {0}
			
			for j in range(L - 1, -1, -1):
				dp_j = set()
				for k in range(j + 1, L + 1):
					if s[j] == '0' and k > j + 1:
						continue
					num = int(s[j:k])
					if num > i:
						continue
					for x in dp[k]:
						total_here = num + x
						if total_here <= i:
							dp_j.add(total_here)
				dp[j] = dp_j
			
			if i in dp[0]:
				total += sq
				
		return total