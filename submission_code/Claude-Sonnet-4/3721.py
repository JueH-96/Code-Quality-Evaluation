class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        # Initialize mentions count for each user
        mentions = [0] * numberOfUsers
        
        # Track offline periods for each user: list of (start_time, end_time) tuples
        offline_periods = [[] for _ in range(numberOfUsers)]
        
        # Sort events by timestamp to process in chronological order
        events.sort(key=lambda x: int(x[1]))
        
        # Process events
        for event in events:
            event_type = event[0]
            timestamp = int(event[1])
            
            if event_type == "OFFLINE":
                user_id = int(event[2])
                # User goes offline for 60 time units
                offline_periods[user_id].append((timestamp, timestamp + 60))
            
            elif event_type == "MESSAGE":
                mentions_string = event[2]
                
                if mentions_string == "ALL":
                    # Mention all users
                    for i in range(numberOfUsers):
                        mentions[i] += 1
                
                elif mentions_string == "HERE":
                    # Mention all online users
                    for i in range(numberOfUsers):
                        if self.is_online(i, timestamp, offline_periods):
                            mentions[i] += 1
                
                else:
                    # Parse individual user mentions
                    tokens = mentions_string.split()
                    for token in tokens:
                        if token.startswith("id"):
                            user_id = int(token[2:])  # Extract number after "id"
                            mentions[user_id] += 1
        
        return mentions
    
    def is_online(self, user_id, timestamp, offline_periods):
        # Check if user is online at given timestamp
        for start_time, end_time in offline_periods[user_id]:
            if start_time <= timestamp < end_time:
                return False
        return True