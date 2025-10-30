class Solution:
	def waysToReachStair(self, k: int) -> int:
		n = 0
		ans = 0
		while True:
			power = 1 << n
			if power < k:
				n += 1
				continue
			d = power - k
			if d > n + 1:
				break
			r = d
			if r > n + 1 - r:
				r = n + 1 - r
			num = 1
			for i in range(1, r + 1):
				num = num * (n + 1 - i + 1) // i
			ans += num
			n += 1
		return ans