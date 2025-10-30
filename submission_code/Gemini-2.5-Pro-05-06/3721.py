from typing import List

class Solution:
  def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
    
    mentions = [0] * numberOfUsers
    # offline_until[user_id] stores the timestamp T when user_id becomes online again.
    # A user is considered offline if current_time < offline_until[user_id].
    # They are online if current_time >= offline_until[user_id].
    # Initially all users are online. We represent this by setting offline_until[user_id] = 0.
    # Given timestamps are positive integers (>=1), 0 is effectively a time in the "distant past"
    # relative to event timestamps, ensuring users are initially online.
    offline_until = [0] * numberOfUsers
    
    # Transform events into a sortable format: (timestamp, type_priority, original_event_data)
    # type_priority: 0 for OFFLINE, 1 for MESSAGE.
    # This is crucial: status changes (OFFLINE events making users offline, or users automatically
    # coming back online) must be resolved before MESSAGE events at the same timestamp are processed.
    # Sorting by (timestamp, type_priority) ensures OFFLINE events are handled first for a given timestamp.
    # Users coming back online is handled by checking `current_time >= offline_until[user_id]`
    # which correctly reflects their status at current_time.
    
    processed_events = []
    for event_item in events: # Using event_item to avoid potential name clashes
        timestamp = int(event_item[1])
        event_type = event_item[0]
        
        if event_type == "OFFLINE":
            # OFFLINE events get lower priority (0) to be processed first at the same timestamp.
            processed_events.append((timestamp, 0, event_item))
        elif event_type == "MESSAGE":
            # MESSAGE events get higher priority (1).
            processed_events.append((timestamp, 1, event_item))
            
    # Sort events:
    # 1. Primarily by timestamp (ascending).
    # 2. Secondarily by type_priority (ascending, so 0 (OFFLINE) comes before 1 (MESSAGE)).
    # Python's default sort for tuples works lexicographically, achieving this.
    processed_events.sort()
    
    for timestamp, type_priority, event_data in processed_events:
        current_time = timestamp # The time at which this event occurs.
        
        # Handle OFFLINE event
        if type_priority == 0: # This is an OFFLINE event
            # event_data structure: ["OFFLINE", "timestamp_string", "user_id_string"]
            user_id_str = event_data[2]
            user_id = int(user_id_str)
            
            # The user user_id becomes offline at current_time for 60 time units.
            # They will be online again at current_time + 60.
            # Thus, they are offline during the interval [current_time, current_time + 60).
            offline_until[user_id] = current_time + 60
        
        # Handle MESSAGE event
        else: # This is a MESSAGE event (type_priority == 1)
            # event_data structure: ["MESSAGE", "timestamp_string", "mentions_string"]
            mentions_string = event_data[2]
            
            if mentions_string == "ALL":
                # Mention all users, regardless of their online/offline status.
                for i in range(numberOfUsers):
                    mentions[i] += 1
            elif mentions_string == "HERE":
                # Mention all users who are currently online.
                for i in range(numberOfUsers):
                    # A user i is online if current_time >= offline_until[i].
                    # This check correctly reflects status for users:
                    #  - who were never offline (offline_until[i] == 0).
                    #  - whose offline period ended before current_time (offline_until[i] < current_time).
                    #  - whose offline period ends exactly at current_time (offline_until[i] == current_time).
                    #    (They become online AT current_time, so are included).
                    if current_time >= offline_until[i]:
                        mentions[i] += 1
            else: 
                # mentions_string is a space-separated list of "id<number>" tokens (e.g., "id0 id1 id0").
                # These mentions count regardless of users' online/offline status.
                # Each mention counts, even if a user is mentioned multiple times in one string.
                individual_mention_tokens = mentions_string.split(' ')
                for token in individual_mention_tokens:
                    # A token is like "id<number>", e.g., "id0", "id15".
                    # Extract <number> by slicing off the "id" prefix (first 2 characters).
                    user_id = int(token[2:]) 
                    mentions[user_id] += 1
                    
    return mentions