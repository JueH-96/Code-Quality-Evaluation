class Solution:
	def countSymmetricIntegers(self, low: int, high: int) -> int:
		count = 0
		for num in range(low, high + 1):
			s = str(num)
			n = len(s)
			if n % 2 != 0:
				continue
			half = n // 2
			first_sum = sum(int(d) for d in s[:half])
			second_sum = sum(int(d) for d in s[half:])
			if first_sum == second_sum:
				count += 1
		return count