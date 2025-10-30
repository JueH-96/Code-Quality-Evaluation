class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        online = [True] * numberOfUsers  # All users start online
        offline_until = [0] * numberOfUsers  # When each user comes back online
        
        # Sort events by timestamp, with OFFLINE events before MESSAGE events at the same timestamp
        events.sort(key=lambda x: (int(x[1]), x[0] == "MESSAGE"))
        
        for event in events:
            timestamp = int(event[1])
            
            # First, update online status (users coming back online automatically)
            for user_id in range(numberOfUsers):
                if offline_until[user_id] <= timestamp:
                    online[user_id] = True
            
            if event[0] == "OFFLINE":
                user_id = int(event[2])
                online[user_id] = False
                offline_until[user_id] = timestamp + 60
            
            elif event[0] == "MESSAGE":
                mentions_string = event[2]
                
                if mentions_string == "ALL":
                    # Mention all users (regardless of online status)
                    for user_id in range(numberOfUsers):
                        mentions[user_id] += 1
                
                elif mentions_string == "HERE":
                    # Mention all online users only
                    for user_id in range(numberOfUsers):
                        if online[user_id]:
                            mentions[user_id] += 1
                
                else:
                    # Parse individual id mentions
                    tokens = mentions_string.split()
                    for token in tokens:
                        if token.startswith("id"):
                            user_id = int(token[2:])  # Remove "id" prefix
                            mentions[user_id] += 1
        
        return mentions