class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        # Initialize mentions count array and offline status tracking
        mentions = [0] * numberOfUsers
        offline_until = [0] * numberOfUsers
        
        # Sort events by timestamp
        events.sort(key=lambda x: int(x[1]))
        
        for event in events:
            event_type, timestamp, data = event
            timestamp = int(timestamp)
            
            if event_type == "OFFLINE":
                user_id = int(data)
                offline_until[user_id] = timestamp + 60
            
            elif event_type == "MESSAGE":
                # Process mentions
                if data == "ALL":
                    # Mention all users
                    for i in range(numberOfUsers):
                        mentions[i] += 1
                
                elif data == "HERE":
                    # Mention only online users
                    for i in range(numberOfUsers):
                        if offline_until[i] <= timestamp:
                            mentions[i] += 1
                
                else:
                    # Process individual mentions
                    mention_ids = data.split()
                    for mention in mention_ids:
                        user_id = int(mention[2:])  # Remove 'id' prefix
                        mentions[user_id] += 1
        
        return mentions