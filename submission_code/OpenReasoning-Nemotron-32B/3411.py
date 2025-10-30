class Solution:
	def popcount_sum(self, n):
		if n <= 0:
			return 0
		total = 0
		while n:
			k = n.bit_length() - 1
			b = 1 << k
			total += k * (b // 2) + (n - b + 1)
			n -= b
		return total

	def find_block(self, i):
		if i < 0:
			return 0
		low, high = 0, i + 1
		while low < high:
			mid = (low + high) // 2
			if self.popcount_sum(mid) <= i:
				low = mid + 1
			else:
				high = mid
		return low

	def get_exponent_in_block(self, n, pos):
		temp = n
		count = 0
		exp = 0
		while temp:
			if temp & 1:
				if count == pos:
					return exp
				count += 1
			temp //= 2
			exp += 1
		return -1

	def count_set_upto(self, x, k):
		if x < 0:
			return 0
		cycle = 1 << (k + 1)
		full_cycles = (x + 1) // cycle
		rem = (x + 1) % cycle
		return full_cycles * (1 << k) + max(0, rem - (1 << k))

	def count_set_in_range(self, low, high, k):
		if low > high:
			return 0
		return self.count_set_upto(high, k) - self.count_set_upto(low - 1, k)

	def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
		ans = []
		for q in queries:
			L, R, mod = q
			nL = self.find_block(L)
			startL = self.popcount_sum(nL - 1)
			posL = L - startL
			
			nR = self.find_block(R)
			startR = self.popcount_sum(nR - 1)
			posR = R - startR
			
			if nL == nR:
				total_exponent = 0
				len_block = bin(nL).count("1")
				for j in range(posL, posR + 1):
					total_exponent += self.get_exponent_in_block(nL, j)
			else:
				total_exponent = 0
				lenL = bin(nL).count("1")
				for j in range(posL, lenL):
					total_exponent += self.get_exponent_in_block(nL, j)
				
				if nL + 1 <= nR - 1:
					for k in range(0, 61):
						cnt = self.count_set_in_range(nL + 1, nR - 1, k)
						total_exponent += k * cnt
				
				for j in range(0, posR + 1):
					total_exponent += self.get_exponent_in_block(nR, j)
			
			result = pow(2, total_exponent, mod)
			ans.append(result)
			
		return ans