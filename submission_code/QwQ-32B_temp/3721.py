from typing import List

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        # Sort events by timestamp and type (OFFLINE before MESSAGE at same time)
        sorted_events = sorted(events, key=lambda x: (int(x[1]), 0 if x[0] == "OFFLINE" else 1))
        
        mentions = [0] * numberOfUsers
        next_online = [0] * numberOfUsers  # Initially all users are online
        
        for event in sorted_events:
            event_type, timestamp_str, rest = event
            timestamp = int(timestamp_str)
            
            if event_type == "OFFLINE":
                user_id = int(rest)
                next_online[user_id] = timestamp + 60
            else:  # MESSAGE event
                mentions_str = rest
                if mentions_str == "ALL":
                    for u in range(numberOfUsers):
                        mentions[u] += 1
                elif mentions_str == "HERE":
                    for u in range(numberOfUsers):
                        if next_online[u] <= timestamp:
                            mentions[u] += 1
                else:
                    tokens = mentions_str.split()
                    for token in tokens:
                        user_id = int(token[2:])  # Extract the number part after 'id'
                        mentions[user_id] += 1
        
        return mentions