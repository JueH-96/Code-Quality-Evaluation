MOD1 = 10**9 + 7
MOD2 = 10**9 + 9
base1 = 131
base2 = 1313

class Solution:
	def minStartingIndex(self, s: str, pattern: str) -> int:
		n = len(pattern)
		m = len(s)
		if n == 0:
			return 0
		max_len = max(n, m)
		
		pow1 = [1] * (max_len + 1)
		pow2 = [1] * (max_len + 1)
		for i in range(1, max_len + 1):
			pow1[i] = (pow1[i-1] * base1) % MOD1
			pow2[i] = (pow2[i-1] * base2) % MOD2
		
		H_p1 = [0] * (n + 1)
		H_p2 = [0] * (n + 1)
		for i in range(n):
			char_val = ord(pattern[i]) - ord('a') + 1
			H_p1[i+1] = (H_p1[i] * base1 + char_val) % MOD1
			H_p2[i+1] = (H_p2[i] * base2 + char_val) % MOD2
		
		H_s1 = [0] * (m + 1)
		H_s2 = [0] * (m + 1)
		for i in range(m):
			char_val = ord(s[i]) - ord('a') + 1
			H_s1[i+1] = (H_s1[i] * base1 + char_val) % MOD1
			H_s2[i+1] = (H_s2[i] * base2 + char_val) % MOD2
		
		for i in range(0, m - n + 1):
			hash_s1_win = (H_s1[i+n] - H_s1[i] * pow1[n]) % MOD1
			hash_s2_win = (H_s2[i+n] - H_s2[i] * pow2[n]) % MOD2
			if hash_s1_win == H_p1[n] and hash_s2_win == H_p2[n]:
				return i
				
			low, high = 0, n - 1
			while low < high:
				mid = (low + high) // 2
				len_seg = mid + 1
				hash_s1_seg = (H_s1[i+mid+1] - H_s1[i] * pow1[mid+1]) % MOD1
				hash_s2_seg = (H_s2[i+mid+1] - H_s2[i] * pow2[mid+1]) % MOD2
				hash_p1_seg = H_p1[mid+1]
				hash_p2_seg = H_p2[mid+1]
				
				if hash_s1_seg == hash_p1_seg and hash_s2_seg == hash_p2_seg:
					low = mid + 1
				else:
					high = mid
					
			l = low
			rest_len = n - l - 1
			if rest_len == 0:
				return i
			else:
				hash_s1_rest = (H_s1[i+n] - H_s1[i+l+1] * pow1[rest_len]) % MOD1
				hash_s2_rest = (H_s2[i+n] - H_s2[i+l+1] * pow2[rest_len]) % MOD2
				hash_p1_rest = (H_p1[n] - H_p1[l+1] * pow1[rest_len]) % MOD1
				hash_p2_rest = (H_p2[n] - H_p2[l+1] * pow2[rest_len]) % MOD2
				
				if hash_s1_rest == hash_p1_rest and hash_s2_rest == hash_p2_rest:
					return i
					
		return -1