from typing import List

class Solution:
	def beautifulSplits(self, nums: List[int]) -> int:
		n = len(nums)
		if n < 3:
			return 0
		
		base1, base2 = 131, 1331
		mod1, mod2 = 10**9 + 7, 10**9 + 9
		
		pow1 = [1] * (n + 1)
		pow2 = [1] * (n + 1)
		for i in range(1, n + 1):
			pow1[i] = (pow1[i - 1] * base1) % mod1
			pow2[i] = (pow2[i - 1] * base2) % mod2
		
		h1 = [0] * (n + 1)
		h2 = [0] * (n + 1)
		for i in range(1, n + 1):
			h1[i] = (h1[i - 1] * base1 + nums[i - 1]) % mod1
			h2[i] = (h2[i - 1] * base2 + nums[i - 1]) % mod2
		
		total = 0
		for i in range(1, n - 1):
			for j in range(i + 1, n):
				if j - i >= i:
					if i + i <= n:
						H1_mod1 = (h1[i + i] - h1[i] * pow1[i]) % mod1
						if H1_mod1 < 0:
							H1_mod1 += mod1
						H1_mod2 = (h2[i + i] - h2[i] * pow2[i]) % mod2
						if H1_mod2 < 0:
							H1_mod2 += mod2
						if H1_mod1 == h1[i] and H1_mod2 == h2[i]:
							total += 1
							continue
				if n - j >= j - i:
					L = j - i
					H2_mod1 = (h1[j] - h1[i] * pow1[L]) % mod1
					if H2_mod1 < 0:
						H2_mod1 += mod1
					H2_mod2 = (h2[j] - h2[i] * pow2[L]) % mod2
					if H2_mod2 < 0:
						H2_mod2 += mod2
					H3_mod1 = (h1[j + L] - h1[j] * pow1[L]) % mod1
					if H3_mod1 < 0:
						H3_mod1 += mod1
					H3_mod2 = (h2[j + L] - h2[j] * pow2[L]) % mod2
					if H3_mod2 < 0:
						H3_mod2 += mod2
					if H2_mod1 == H3_mod1 and H2_mod2 == H3_mod2:
						total += 1
		return total