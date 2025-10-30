import collections

class Solution:
	def findWinningPlayer(self, skills: List[int], k: int) -> int:
		n = len(skills)
		max_skill = max(skills)
		max_index = skills.index(max_skill)
		if k >= n:
			return max_index
		
		q = collections.deque(range(n))
		current_winner = q.popleft()
		current_streak = 0
		
		while True:
			next_player = q.popleft()
			if skills[current_winner] > skills[next_player]:
				current_streak += 1
				q.append(next_player)
			else:
				q.append(current_winner)
				current_winner = next_player
				current_streak = 1
				
			if current_streak == k or current_winner == max_index:
				return current_winner