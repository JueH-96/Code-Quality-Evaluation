class Solution:
	def minStartingIndex(self, s: str, pattern: str) -> int:
		n = len(s)
		m = len(pattern)
		total_str = pattern + '$' + s
		total_len = len(total_str)
		
		Z = [0] * total_len
		l, r = 0, 0
		for i in range(1, total_len):
			if i <= r:
				Z[i] = min(r - i + 1, Z[i - l])
			while i + Z[i] < total_len and total_str[Z[i]] == total_str[i + Z[i]]:
				Z[i] += 1
			if i + Z[i] - 1 > r:
				l = i
				r = i + Z[i] - 1
		
		base = 131
		mod1 = 10**9 + 7
		mod2 = 10**9 + 9
		
		def build_hash(text, mod):
			n_text = len(text)
			h = [0] * (n_text + 1)
			pow_arr = [1] * (n_text + 1)
			for i in range(1, n_text + 1):
				char_val = ord(text[i-1]) - ord('a') + 1
				h[i] = (h[i-1] * base + char_val) % mod
				pow_arr[i] = (pow_arr[i-1] * base) % mod
			return h, pow_arr
		
		hash1_s, pow1_s = build_hash(s, mod1)
		hash2_s, pow2_s = build_hash(s, mod2)
		hash1_p, pow1_p = build_hash(pattern, mod1)
		hash2_p, pow2_p = build_hash(pattern, mod2)
		
		def get_hash_val(h, pow_arr, l, r, mod):
			length = r - l + 1
			product = (h[l] * pow_arr[length]) % mod
			hash_val = (h[r+1] - product) % mod
			if hash_val < 0:
				hash_val += mod
			return hash_val
		
		for i in range(0, n - m + 1):
			pos_in_total = len(pattern) + 1 + i
			common1 = Z[pos_in_total]
			
			if common1 == m:
				return i
				
			len_rest = m - common1 - 1
			if len_rest < 0:
				continue
				
			if len_rest == 0:
				return i
			else:
				s_start = i + common1 + 1
				s_end = s_start + len_rest - 1
				p_start = common1 + 1
				p_end = p_start + len_rest - 1
				
				if s_end >= n or p_end >= m:
					continue
					
				s_hash1 = get_hash_val(hash1_s, pow1_s, s_start, s_end, mod1)
				p_hash1 = get_hash_val(hash1_p, pow1_p, p_start, p_end, mod1)
				if s_hash1 != p_hash1:
					continue
					
				s_hash2 = get_hash_val(hash2_s, pow2_s, s_start, s_end, mod2)
				p_hash2 = get_hash_val(hash2_p, pow2_p, p_start, p_end, mod2)
				if s_hash2 != p_hash2:
					continue
					
				return i
				
		return -1