class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        # Initialize mention counts
        mentions = [0] * numberOfUsers
        
        # Track offline users and when they come back online
        offline_until = {}  # user_id -> timestamp when they come back online
        
        # Sort events by timestamp, then by type (OFFLINE before MESSAGE)
        def event_key(event):
            timestamp = int(event[1])
            # OFFLINE = 0, MESSAGE = 1
            type_priority = 0 if event[0] == "OFFLINE" else 1
            return (timestamp, type_priority)
        
        events.sort(key=event_key)
        
        for event in events:
            event_type, timestamp_str, data = event
            timestamp = int(timestamp_str)
            
            # Update online/offline status
            # Remove users who have come back online
            offline_until = {uid: come_back_time 
                           for uid, come_back_time in offline_until.items() 
                           if come_back_time > timestamp}
            
            if event_type == "OFFLINE":
                user_id = int(data)
                offline_until[user_id] = timestamp + 60
            
            elif event_type == "MESSAGE":
                if data == "ALL":
                    for i in range(numberOfUsers):
                        mentions[i] += 1
                
                elif data == "HERE":
                    for i in range(numberOfUsers):
                        if i not in offline_until:
                            mentions[i] += 1
                
                else:
                    # Parse individual mentions
                    for token in data.split():
                        if token.startswith("id"):
                            user_id = int(token[2:])
                            mentions[user_id] += 1
        
        return mentions