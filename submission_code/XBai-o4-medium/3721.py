class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        offline_until = [0] * numberOfUsers
        
        # Sort events by timestamp, with OFFLINE events before MESSAGE events at the same time
        sorted_events = sorted(events, key=lambda x: (int(x[1]), 0 if x[0] == "OFFLINE" else 1))
        
        for event in sorted_events:
            if event[0] == "OFFLINE":
                timestamp = int(event[1])
                user_id = int(event[2])
                offline_until[user_id] = timestamp + 60
            else:  # MESSAGE event
                timestamp = int(event[1])
                mention_str = event[2]
                if mention_str == "ALL":
                    for user in range(numberOfUsers):
                        mentions[user] += 1
                elif mention_str == "HERE":
                    for user in range(numberOfUsers):
                        if offline_until[user] <= timestamp:
                            mentions[user] += 1
                else:
                    parts = mention_str.split()
                    for part in parts:
                        if part.startswith("id"):
                            user_id = int(part[2:])
                            mentions[user_id] += 1
        
        return mentions