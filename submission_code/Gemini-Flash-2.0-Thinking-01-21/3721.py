from typing import List

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        """
        Counts the number of times each user is mentioned in MESSAGE events,
        considering user online/offline status changes due to OFFLINE events.

        Args:
            numberOfUsers: The total number of users (0 to numberOfUsers - 1).
            events: A list of event records [type, timestamp_str, data].

        Returns:
            A list where mentions[i] is the total mention count for user i.
        """
        # Initialize mentions count for each user
        mentions = [0] * numberOfUsers

        # Initialize online status for each user (initially all online)
        online_status = [True] * numberOfUsers

        # Prepare events for sorting, including implied online events
        # We'll use a tuple (timestamp, priority, data)
        # priority: -1 for implied online (highest priority), 0 for OFFLINE, 1 for MESSAGE (lowest priority at same timestamp)
        # data for implied online is just the user ID
        # data for original events is the original event list
        processed_events = []

        for event in events:
            event_type, timestamp_str, data = event
            timestamp = int(timestamp_str)

            if event_type == "MESSAGE":
                # Message events have priority 1
                processed_events.append((timestamp, 1, event))
            elif event_type == "OFFLINE":
                user_id = int(data)
                # OFFLINE events have priority 0
                processed_events.append((timestamp, 0, event))
                # Add implied online event 60 time units later
                # Implied online events have priority -1 (processed first at a given timestamp)
                # The user_id is passed as data for the implied online event
                processed_events.append((timestamp + 60, -1, user_id))

        # Sort events by timestamp, then by priority
        # The default sort for tuples works correctly: sorts by first element, then second, etc.
        processed_events.sort()

        # Process events in sorted order
        for timestamp, priority, data in processed_events:

            if priority == -1: # Implied ONLINE event
                user_id = data # User ID is stored directly in data
                online_status[user_id] = True

            elif priority == 0: # OFFLINE event
                # The original event data is stored in `data`
                original_event = data
                event_type, timestamp_str, user_id_str = original_event
                user_id = int(user_id_str)
                # Set status to offline
                online_status[user_id] = False

            elif priority == 1: # MESSAGE event
                # The original event data is stored in `data`
                original_event = data
                event_type, timestamp_str, mentions_string = original_event

                if mentions_string == "ALL":
                    # Mention all users (online or offline)
                    for i in range(numberOfUsers):
                        mentions[i] += 1

                elif mentions_string == "HERE":
                    # Mention only online users
                    for i in range(numberOfUsers):
                        if online_status[i]:
                            mentions[i] += 1

                else: # Contains id<number> tokens, e.g., "id0 id1 id0"
                    # Split the string by whitespace and process each id<number> token
                    id_tokens = mentions_string.split()
                    for token in id_tokens:
                        # Extract the user ID from the token "id<number>"
                        # Assumes token starts with "id" followed by digits
                        # Convert to integer
                        user_id = int(token[2:]) # Skip "id" prefix

                        # Mention the user (even if offline)
                        # The problem asks to count each mention separately
                        mentions[user_id] += 1

        return mentions