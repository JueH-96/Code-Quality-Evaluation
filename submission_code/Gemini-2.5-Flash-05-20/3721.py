from typing import List, Dict

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        is_online = [True] * numberOfUsers
        # offline_expiry: maps user_id to the timestamp when they become online again
        # This dictionary stores users who are currently offline and their re-online time.
        offline_expiry: Dict[int, int] = {} 

        # Sort events:
        # Primary sort key: timestamp (ascending).
        # Secondary sort key: event type priority. OFFLINE events (priority 0)
        # must be processed before MESSAGE events (priority 1) if they share
        # the same timestamp. This is crucial for handling status changes
        # before message processing at the same time.
        events.sort(key=lambda x: (int(x[1]), 0 if x[0] == "OFFLINE" else 1))

        for event_type, timestamp_str, data_str in events:
            current_timestamp = int(timestamp_str)

            # --- Step 1: Process users coming online whose offline period has expired ---
            # Create a list of users to be brought online. This is done to safely modify
            # `offline_expiry` dictionary (deleting items) while iterating over it.
            users_to_bring_online = []
            # Using list(offline_expiry.items()) creates a copy of the items,
            # allowing safe iteration while modifications occur on the original dict.
            for user_id, expiry_time in list(offline_expiry.items()):
                if expiry_time <= current_timestamp:
                    is_online[user_id] = True  # Set user status to online
                    users_to_bring_online.append(user_id)
            
            # Remove users who just came online from the offline_expiry tracking
            for user_id in users_to_bring_online:
                del offline_expiry[user_id]

            # --- Step 2: Process the current event ---
            if event_type == "OFFLINE":
                user_id = int(data_str)
                is_online[user_id] = False  # Set user status to offline
                # Record when this user will automatically come back online
                offline_expiry[user_id] = current_timestamp + 60
            elif event_type == "MESSAGE":
                if data_str == "ALL":
                    # All users are mentioned, regardless of online/offline status
                    for i in range(numberOfUsers):
                        mentions[i] += 1
                elif data_str == "HERE":
                    # Only currently online users are mentioned
                    for i in range(numberOfUsers):
                        if is_online[i]:
                            mentions[i] += 1
                else: # Specific IDs like "id0 id1 id2..."
                    # Split the mentions string by space to get individual ID tokens
                    mentioned_tokens = data_str.split()
                    for token in mentioned_tokens:
                        # Extract the integer user ID from the "id<number>" string
                        # e.g., "id0" -> 0, "id12" -> 12
                        user_id = int(token[2:]) 
                        # Increment mention count for the specific user
                        # Note: Offline users can be mentioned this way.
                        mentions[user_id] += 1
        
        return mentions