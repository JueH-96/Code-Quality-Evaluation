class Solution:
	def answerString(self, word: str, numFriends: int) -> str:
		n = len(word)
		L_val = n - (numFriends - 1)
		candidate_start = 0
		candidate_len = min(L_val, n)
		
		for i in range(1, n):
			L_i = min(L_val, n - i)
			j = 0
			while j < min(L_i, candidate_len) and word[i+j] == word[candidate_start+j]:
				j += 1
				
			if j < min(L_i, candidate_len):
				if word[i+j] > word[candidate_start+j]:
					candidate_start = i
					candidate_len = L_i
			else:
				if L_i > candidate_len:
					candidate_start = i
					candidate_len = L_i
					
		return word[candidate_start:candidate_start+candidate_len]