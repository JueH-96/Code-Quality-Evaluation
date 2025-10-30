class Solution:
	def countCompleteSubstrings(self, word: str, k: int) -> int:
		n = len(word)
		if n == 0:
			return 0
		
		bad = []
		for i in range(n-1):
			diff = abs(ord(word[i]) - ord(word[i+1]))
			if diff > 2:
				bad.append(1)
			else:
				bad.append(0)
				
		B = [0] * n
		for i in range(1, n):
			B[i] = B[i-1] + bad[i-1]
			
		total_count = 0
		for d in range(1, 27):
			L = d * k
			if L > n:
				break
				
			freq = [0] * 26
			distinct_count = 0
			ok = 0
			
			for j in range(L):
				c = word[j]
				idx = ord(c) - ord('a')
				old_count = freq[idx]
				freq[idx] += 1
				if old_count == 0:
					distinct_count += 1
				if freq[idx] == k:
					ok += 1
				if old_count == k:
					ok -= 1
					
			if distinct_count == d and ok == d and (B[L-1] - B[0] == 0):
				total_count += 1
				
			for i in range(1, n - L + 1):
				c_remove = word[i-1]
				idx_remove = ord(c_remove) - ord('a')
				old_count_remove = freq[idx_remove]
				freq[idx_remove] -= 1
				if old_count_remove == 1:
					distinct_count -= 1
				if old_count_remove == k:
					ok -= 1
				if freq[idx_remove] == k:
					ok += 1
					
				c_add = word[i+L-1]
				idx_add = ord(c_add) - ord('a')
				old_count_add = freq[idx_add]
				freq[idx_add] += 1
				if old_count_add == 0:
					distinct_count += 1
				if old_count_add == k:
					ok -= 1
				if freq[idx_add] == k:
					ok += 1
					
				if distinct_count == d and ok == d and (B[i+L-1] - B[i] == 0):
					total_count += 1
					
		return total_count