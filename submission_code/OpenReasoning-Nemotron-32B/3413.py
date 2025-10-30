from collections import deque

class Solution:
	def findWinningPlayer(self, skills: List[int], k: int) -> int:
		n = len(skills)
		if k >= n:
			max_skill = max(skills)
			for i in range(n):
				if skills[i] == max_skill:
					return i
		
		q = deque(range(n))
		a = q.popleft()
		b = q.popleft()
		if skills[a] > skills[b]:
			winner = a
			q.append(b)
		else:
			winner = b
			q.append(a)
		streak = 1
		
		if streak == k:
			return winner
			
		while streak < k:
			challenger = q.popleft()
			if skills[winner] > skills[challenger]:
				streak += 1
				q.append(challenger)
			else:
				q.append(winner)
				winner = challenger
				streak = 1
		return winner