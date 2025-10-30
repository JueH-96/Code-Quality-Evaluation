from typing import List

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        # This solution processes events in chronological order, updating user statuses
        # and mention counts as it goes.

        # 1. Initialize data structures
        # `mentions` will store the final count for each user.
        mentions = [0] * numberOfUsers
        
        # `offline_until[i]` stores the timestamp at which user `i` becomes online again.
        # A value of 0 indicates the user is online from the start.
        offline_until = [0] * numberOfUsers

        # 2. Sort events to process them in the correct order.
        # The primary sort key is the timestamp (as an integer).
        # The secondary sort key ensures that for a given timestamp, OFFLINE events
        # are processed before MESSAGE events. This is crucial for correctly
        # determining who is online when a "HERE" message is sent.
        # `e[0] == "MESSAGE"` evaluates to False (0) for "OFFLINE" and True (1) for "MESSAGE",
        # achieving the desired order.
        events.sort(key=lambda e: (int(e[1]), e[0] == "MESSAGE"))

        # 3. Iterate through the sorted events and process them.
        for event_type, timestamp_str, data_str in events:
            timestamp = int(timestamp_str)

            if event_type == "OFFLINE":
                # An OFFLINE event marks a user as offline for 60 time units.
                user_id = int(data_str)
                # The user will be online again at `timestamp + 60`.
                offline_until[user_id] = timestamp + 60
            
            elif event_type == "MESSAGE":
                # A MESSAGE event increments mention counts for specified users.
                if data_str == "ALL":
                    # Mention all users, regardless of their online status.
                    for i in range(numberOfUsers):
                        mentions[i] += 1
                elif data_str == "HERE":
                    # Mention all users who are currently online.
                    for i in range(numberOfUsers):
                        # A user is considered online at `timestamp` if `timestamp` is
                        # greater than or equal to the time they are set to come back online.
                        if timestamp >= offline_until[i]:
                            mentions[i] += 1
                else:
                    # Mention specific users from a space-separated string of IDs.
                    # e.g., "id0 id1 id0"
                    id_tokens = data_str.split(' ')
                    for token in id_tokens:
                        # Parse "id<number>" to get the user ID.
                        user_id = int(token[2:])
                        mentions[user_id] += 1
        
        # 4. Return the final list of mention counts.
        return mentions