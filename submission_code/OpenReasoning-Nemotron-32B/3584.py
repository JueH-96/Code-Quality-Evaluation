from typing import List

class Solution:
	def validSequence(self, word1: str, word2: str) -> List[int]:
		n1 = len(word1)
		n2 = len(word2)
		if n2 == 0:
			return []
		
		next_arr = [[n1] * (n1 + 1) for _ in range(26)]
		for c_idx in range(26):
			next_arr[c_idx][n1] = n1
			for i in range(n1 - 1, -1, -1):
				if ord(word1[i]) - ord('a') == c_idx:
					next_arr[c_idx][i] = i
				else:
					next_arr[c_idx][i] = next_arr[c_idx][i + 1]
		
		dp0 = [n1] * n2
		c0 = ord(word2[0]) - ord('a')
		dp0[0] = next_arr[c0][0]
		
		if n2 == 1:
			if dp0[0] < n1:
				return [dp0[0]]
		
		dp1 = [n1] * n2
		dp1[0] = 0
		from1 = [-1] * n2
		
		for i in range(1, n2):
			c = ord(word2[i]) - ord('a')
			if dp0[i - 1] < n1:
				next_pos = dp0[i - 1] + 1
				if next_pos < n1:
					dp0[i] = next_arr[c][next_pos]
				else:
					dp0[i] = n1
			else:
				dp0[i] = n1
			
			option1 = n1
			if dp1[i - 1] < n1:
				next_pos = dp1[i - 1] + 1
				if next_pos < n1:
					option1 = next_arr[c][next_pos]
				else:
					option1 = n1
			else:
				option1 = n1
			
			option2 = n1
			if dp0[i - 1] < n1:
				next_pos = dp0[i - 1] + 1
				if next_pos < n1:
					option2 = next_pos
				else:
					option2 = n1
			else:
				option2 = n1
			
			if option1 < option2:
				dp1[i] = option1
				from1[i] = 1
			else:
				dp1[i] = option2
				from1[i] = 0
		
		if dp0[n2 - 1] < n1:
			return dp0
		
		if dp1[n2 - 1] < n1:
			res = [0] * n2
			res[n2 - 1] = dp1[n2 - 1]
			current_state = 1
			for i in range(n2 - 1, 0, -1):
				if current_state == 1:
					if from1[i] == 1:
						res[i - 1] = dp1[i - 1]
						current_state = 1
					else:
						res[i - 1] = dp0[i - 1]
						current_state = 0
				else:
					res[i - 1] = dp0[i - 1]
					current_state = 0
			return res
		
		return []