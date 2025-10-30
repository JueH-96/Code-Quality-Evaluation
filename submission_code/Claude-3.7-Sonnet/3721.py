class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        offline_times = {}  # Maps user_id to the timestamp they'll be back online
        
        # Sort events by timestamp, then by type (OFFLINE before MESSAGE)
        events.sort(key=lambda x: (int(x[1]), 0 if x[0] == "OFFLINE" else 1))
        
        for event_type, timestamp, data in events:
            timestamp = int(timestamp)
            
            # Update user's online status
            for user_id in list(offline_times.keys()):
                if timestamp >= offline_times[user_id]:
                    offline_times.pop(user_id)
            
            if event_type == "OFFLINE":
                user_id = int(data)
                offline_times[user_id] = timestamp + 60
            
            elif event_type == "MESSAGE":
                if data == "ALL":
                    # Mention all users
                    for user_id in range(numberOfUsers):
                        mentions[user_id] += 1
                
                elif data == "HERE":
                    # Mention all online users
                    for user_id in range(numberOfUsers):
                        if user_id not in offline_times:
                            mentions[user_id] += 1
                
                else:
                    # Process specific user mentions
                    user_ids = data.split()
                    for user_id_str in user_ids:
                        user_id = int(user_id_str[2:])  # Remove "id" prefix
                        mentions[user_id] += 1
        
        return mentions