class Solution:
	def punishmentNumber(self, n: int) -> int:
		total = 0
		for i in range(1, n + 1):
			s = str(i * i)
			m = len(s)
			dp = [set() for _ in range(m + 1)]
			dp[0].add(0)
			for j in range(1, m + 1):
				for k in range(j - 1, -1, -1):
					if s[k] == '0' and k < j - 1:
						continue
					num = int(s[k:j])
					if num > i:
						break
					for x in dp[k]:
						total_sum = x + num
						if total_sum <= i:
							dp[j].add(total_sum)
			if i in dp[m]:
				total += i * i
		return total