class Solution:
	def minimumChairs(self, s: str) -> int:
		current = 0
		max_people = 0
		for event in s:
			if event == 'E':
				current += 1
			else:
				current -= 1
			if current > max_people:
				max_people = current
		return max_people