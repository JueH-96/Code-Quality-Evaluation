class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        # Initialize mention counts for all users
        mentions = [0] * numberOfUsers
        
        # Dictionary to track when users come back online (timestamp -> list of user ids)
        come_online_at = {}
        
        # Set to track currently offline users
        offline_users = set()
        
        # Sort events by timestamp, with OFFLINE events before MESSAGE events at same timestamp
        sorted_events = sorted(events, key=lambda x: (int(x[1]), 0 if x[0] == "OFFLINE" else 1))
        
        for event in sorted_events:
            event_type = event[0]
            timestamp = int(event[1])
            
            # First, check if any users should come back online at this timestamp
            if timestamp in come_online_at:
                for user_id in come_online_at[timestamp]:
                    offline_users.discard(user_id)
                del come_online_at[timestamp]
            
            if event_type == "OFFLINE":
                user_id = int(event[2])
                offline_users.add(user_id)
                # User comes back online after 60 time units
                comeback_time = timestamp + 60
                if comeback_time not in come_online_at:
                    come_online_at[comeback_time] = []
                come_online_at[comeback_time].append(user_id)
                
            elif event_type == "MESSAGE":
                mentions_string = event[2]
                
                if mentions_string == "ALL":
                    # Mention all users (including offline ones)
                    for i in range(numberOfUsers):
                        mentions[i] += 1
                        
                elif mentions_string == "HERE":
                    # Mention only online users
                    for i in range(numberOfUsers):
                        if i not in offline_users:
                            mentions[i] += 1
                            
                else:
                    # Parse individual user mentions
                    tokens = mentions_string.split()
                    for token in tokens:
                        if token.startswith("id"):
                            user_id = int(token[2:])  # Extract number after "id"
                            mentions[user_id] += 1
        
        return mentions