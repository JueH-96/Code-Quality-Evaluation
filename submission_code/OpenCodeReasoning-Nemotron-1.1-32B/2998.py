class Solution:
	def countSymmetricIntegers(self, low: int, high: int) -> int:
		count = 0
		for num in range(low, high + 1):
			s = str(num)
			n = len(s)
			if n % 2 != 0:
				continue
			mid = n // 2
			first_half = s[:mid]
			second_half = s[mid:]
			if sum(int(d) for d in first_half) == sum(int(d) for d in second_half):
				count += 1
		return count