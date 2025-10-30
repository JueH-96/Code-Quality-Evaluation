from typing import List

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        # Initialize mentions count for each user
        mentions = [0] * numberOfUsers
        # Initialize online status: initially all are online
        online = [True] * numberOfUsers
        # We need to process events in order of timestamp, and for same timestamp, OFFLINE before MESSAGE
        # So first sort all events by timestamp, and for same timestamp, OFFLINE comes first
        # To do this, we can sort with a key that ensures OFFLINE is processed before MESSAGE for same timestamp
        # Let's assign OFFLINE a priority of 0 and MESSAGE a priority of 1
        def event_key(event):
            timestamp = int(event[1])
            type_priority = 0 if event[0] == "OFFLINE" else 1
            return (timestamp, type_priority)
        
        sorted_events = sorted(events, key=event_key)
        
        # We also need to keep track of scheduled online times: a min-heap (priority queue) where each entry is (time, user_id)
        import heapq
        scheduled_online = []
        
        for event in sorted_events:
            event_type, timestamp_str, data = event
            timestamp = int(timestamp_str)
            
            # Before processing the event, check scheduled_online for any users that should come online now or earlier
            while scheduled_online and scheduled_online[0][0] <= timestamp:
                time, user_id = heapq.heappop(scheduled_online)
                online[user_id] = True
            
            if event_type == "OFFLINE":
                user_id = int(data)
                # Mark the user as offline and schedule their online time
                online[user_id] = False
                heapq.heappush(scheduled_online, (timestamp + 60, user_id))
            elif event_type == "MESSAGE":
                mentions_str = data
                if mentions_str == "ALL":
                    # All users are mentioned once
                    for user_id in range(numberOfUsers):
                        mentions[user_id] += 1
                elif mentions_str == "HERE":
                    # Only online users are mentioned once
                    for user_id in range(numberOfUsers):
                        if online[user_id]:
                            mentions[user_id] += 1
                else:
                    # Parse the ids in the string
                    parts = mentions_str.split()
                    for part in parts:
                        if part.startswith("id"):
                            user_id = int(part[2:])
                            mentions[user_id] += 1
        
        return mentions