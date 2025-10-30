class Solution:
	def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
		def event_priority(event_type):
			if event_type == "OFFLINE":
				return 0
			else:
				return 1
		
		events.sort(key=lambda x: (int(x[1]), event_priority(x[0])))
		
		offline_expire = [0] * numberOfUsers
		mentions = [0] * numberOfUsers
		
		for event in events:
			if event[0] == "OFFLINE":
				t = int(event[1])
				user_id = int(event[2])
				offline_expire[user_id] = t + 60
			else:
				t = int(event[1])
				s = event[2]
				tokens = s.split()
				for token in tokens:
					if token == "ALL":
						for j in range(numberOfUsers):
							mentions[j] += 1
					elif token == "HERE":
						for j in range(numberOfUsers):
							if offline_expire[j] <= t:
								mentions[j] += 1
					else:
						if token.startswith("id"):
							num_str = token[2:]
							if num_str.isdigit():
								user_id = int(num_str)
								if 0 <= user_id < numberOfUsers:
									mentions[user_id] += 1
		return mentions