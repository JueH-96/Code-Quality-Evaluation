class Solution:
	def minTime(self, skill: List[int], mana: List[int]) -> int:
		n = len(skill)
		m = len(mana)
		if m == 0:
			return 0
		
		F_prev = [0] * n
		
		for j in range(m):
			constraints = [F_prev[0]]
			s = 0
			for i in range(1, n):
				s += skill[i-1]
				constraints.append(F_prev[i] - s * mana[j])
			S = max(constraints)
			
			F_curr = [0] * n
			F_curr[0] = S + skill[0] * mana[j]
			for i in range(1, n):
				F_curr[i] = F_curr[i-1] + skill[i] * mana[j]
			
			F_prev = F_curr
		
		return F_prev[-1]