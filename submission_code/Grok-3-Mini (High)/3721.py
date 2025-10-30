class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        offline_until = [0] * numberOfUsers
        
        # Sort events by timestamp and then by event type (OFFLINE before MESSAGE)
        sorted_events = sorted(events, key=lambda x: (int(x[1]), 0 if x[0] == "OFFLINE" else 1))
        
        for event in sorted_events:
            event_type = event[0]
            timestamp = int(event[1])
            
            if event_type == "OFFLINE":
                user_id = int(event[2])
                offline_until[user_id] = timestamp + 60
            elif event_type == "MESSAGE":
                mentions_string = event[2]
                tokens = mentions_string.split()  # Split the string by whitespace
                
                for token in tokens:
                    if token == "ALL":
                        for u in range(numberOfUsers):
                            mentions[u] += 1
                    elif token == "HERE":
                        for u in range(numberOfUsers):
                            if offline_until[u] <= timestamp:
                                mentions[u] += 1
                    elif token.startswith("id"):
                        num_str = token[2:]  # Extract the number part
                        user_id = int(num_str)
                        mentions[user_id] += 1
        
        return mentions