class Solution:
	def minAnagramLength(self, s: str) -> int:
		n = len(s)
		total_freq = [0] * 26
		for c in s:
			idx = ord(c) - ord('a')
			total_freq[idx] += 1
		
		divisors = []
		i = 1
		while i * i <= n:
			if n % i == 0:
				divisors.append(i)
				if i != n // i:
					divisors.append(n // i)
			i += 1
		divisors.sort()
		
		for d in divisors:
			k = n // d
			valid_div = True
			for count in total_freq:
				if count % k != 0:
					valid_div = False
					break
			if not valid_div:
				continue
				
			target = [count // k for count in total_freq]
			valid_seg = True
			for j in range(k):
				seg_freq = [0] * 26
				start = j * d
				for i in range(start, start + d):
					idx = ord(s[i]) - ord('a')
					seg_freq[idx] += 1
				if seg_freq != target:
					valid_seg = False
					break
					
			if valid_seg:
				return d
				
		return n