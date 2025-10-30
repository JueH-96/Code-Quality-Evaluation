class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        offline_end = [0] * numberOfUsers
        
        events_with_ts = []
        for event in events:
            ts = int(event[1])
            events_with_ts.append((ts, event[0], event[2]))
        
        def event_key(event_tuple):
            ts, etype, _ = event_tuple
            priority = 0 if etype == "OFFLINE" else 1
            return (ts, priority)
        
        events_sorted = sorted(events_with_ts, key=event_key)
        
        for event in events_sorted:
            ts = event[0]
            etype = event[1]
            data = event[2]
            if etype == "OFFLINE":
                user_id = int(data)
                offline_end[user_id] = ts + 60
            else:
                tokens = data.split()
                for token in tokens:
                    if token == "ALL":
                        for i in range(numberOfUsers):
                            mentions[i] += 1
                    elif token == "HERE":
                        for i in range(numberOfUsers):
                            if offline_end[i] <= ts:
                                mentions[i] += 1
                    else:
                        try:
                            user_id = int(token[2:])
                            if 0 <= user_id < numberOfUsers:
                                mentions[user_id] += 1
                        except ValueError:
                            pass
        return mentions