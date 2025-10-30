class Solution:
	def beautifulSplits(self, nums: List[int]) -> int:
		n = len(nums)
		if n < 3:
			return 0
		
		base1, base2 = 131, 137
		mod1, mod2 = 10**9+7, 10**9+9
		
		h1 = [0] * (n+1)
		h2 = [0] * (n+1)
		pow1 = [1] * (n+1)
		pow2 = [1] * (n+1)
		
		for i in range(1, n+1):
			pow1[i] = (pow1[i-1] * base1) % mod1
			pow2[i] = (pow2[i-1] * base2) % mod2
			
		for i in range(n):
			h1[i+1] = (h1[i] * base1 + nums[i]) % mod1
			h2[i+1] = (h2[i] * base2 + nums[i]) % mod2
		
		count = 0
		
		for i in range(1, n-1):
			cond1_possible = False
			if 2 * i <= n:
				H0_1 = (h1[i] - h1[0] * pow1[i]) % mod1
				if H0_1 < 0:
					H0_1 += mod1
				H0_2 = (h2[i] - h2[0] * pow2[i]) % mod2
				if H0_2 < 0:
					H0_2 += mod2
				
				Hi_1 = (h1[2*i] - h1[i] * pow1[i]) % mod1
				if Hi_1 < 0:
					Hi_1 += mod1
				Hi_2 = (h2[2*i] - h2[i] * pow2[i]) % mod2
				if Hi_2 < 0:
					Hi_2 += mod2
				
				if H0_1 == Hi_1 and H0_2 == Hi_2:
					cond1_possible = True
			
			for j in range(i+1, n):
				L2 = j - i
				L3 = n - j
				
				if cond1_possible and j >= 2 * i:
					count += 1
				else:
					if L3 >= L2:
						len_seg = L2
						Hij_1 = (h1[j] - h1[i] * pow1[len_seg]) % mod1
						if Hij_1 < 0:
							Hij_1 += mod1
						Hij_2 = (h2[j] - h2[i] * pow2[len_seg]) % mod2
						if Hij_2 < 0:
							Hij_2 += mod2
						
						end_index = j + len_seg
						Hj_1 = (h1[end_index] - h1[j] * pow1[len_seg]) % mod1
						if Hj_1 < 0:
							Hj_1 += mod1
						Hj_2 = (h2[end_index] - h2[j] * pow2[len_seg]) % mod2
						if Hj_2 < 0:
							Hj_2 += mod2
						
						if Hij_1 == Hj_1 and Hij_2 == Hj_2:
							count += 1
		return count