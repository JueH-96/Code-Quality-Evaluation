from typing import List

class Solution:
	def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
		mentions = [0] * numberOfUsers
		offline_until = [0] * numberOfUsers
		
		events.sort(key=lambda x: (int(x[1]), 0 if x[0] == 'OFFLINE' else 1))
		
		for event in events:
			if event[0] == "OFFLINE":
				t = int(event[1])
				user_id = int(event[2])
				offline_until[user_id] = t + 60
			else:
				t = int(event[1])
				mentions_str = event[2]
				tokens = mentions_str.split()
				for token in tokens:
					if token == "ALL":
						for u in range(numberOfUsers):
							mentions[u] += 1
					elif token == "HERE":
						for u in range(numberOfUsers):
							if offline_until[u] <= t:
								mentions[u] += 1
					else:
						if token.startswith("id"):
							num_str = token[2:]
							try:
								u = int(num_str)
								if 0 <= u < numberOfUsers:
									mentions[u] += 1
							except:
								pass
		return mentions