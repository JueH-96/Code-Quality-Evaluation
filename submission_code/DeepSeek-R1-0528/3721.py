class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        offlineUntil = [-1] * numberOfUsers
        
        events.sort(key=lambda e: (int(e[1]), 0 if e[0] == "OFFLINE" else 1))
        
        current_time = -1
        for event in events:
            ts = int(event[1])
            if ts != current_time:
                for i in range(numberOfUsers):
                    if offlineUntil[i] != -1 and offlineUntil[i] <= ts:
                        offlineUntil[i] = -1
                current_time = ts
            
            if event[0] == "OFFLINE":
                user_id = int(event[2])
                offlineUntil[user_id] = ts + 60
            else:
                mstr = event[2]
                if mstr == "ALL":
                    for i in range(numberOfUsers):
                        mentions[i] += 1
                elif mstr == "HERE":
                    for i in range(numberOfUsers):
                        if offlineUntil[i] == -1:
                            mentions[i] += 1
                else:
                    tokens = mstr.split()
                    for token in tokens:
                        user_id = int(token[2:])
                        mentions[user_id] += 1
        
        return mentions