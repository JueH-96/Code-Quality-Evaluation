from typing import List

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        # Initialize mention counts for each user.
        # mentions[i] will store the total mentions for user with id i.
        mentions = [0] * numberOfUsers

        # Initialize online status for each user.
        # is_online[i] is True if user i is online, False otherwise.
        # All users start online.
        is_online = [True] * numberOfUsers

        # Prepare a list of all events to be processed, including future "online again" events.
        # Each item in all_events will be a tuple: (timestamp, priority, event_type, data)
        # Priority is used to define the processing order for events occurring at the exact same timestamp:
        # OFFLINE events have the highest priority (0), processed first.
        # Generated ONLINE_AGAIN events have medium priority (1), processed second.
        # MESSAGE events have the lowest priority (2), processed last.
        # This order respects the rule that status changes are processed before messages at the same timestamp.
        all_events = []

        # Map event types to their sorting priority.
        priority_map = {
            "OFFLINE": 0,
            "ONLINE_AGAIN": 1,
            "MESSAGE": 2
        }

        # Populate the all_events list from the input events.
        for event in events:
            event_type = event[0]
            timestamp = int(event[1])
            data = event[2]

            if event_type == "MESSAGE":
                mention_string = data
                all_events.append((timestamp, priority_map[event_type], event_type, mention_string))
            elif event_type == "OFFLINE":
                user_id = int(data)
                # Add the original OFFLINE event.
                all_events.append((timestamp, priority_map[event_type], event_type, user_id))
                # Add the corresponding implicit ONLINE_AGAIN event 60 time units later.
                online_again_timestamp = timestamp + 60
                all_events.append((online_again_timestamp, priority_map["ONLINE_AGAIN"], "ONLINE_AGAIN", user_id))

        # Sort all events chronologically.
        # Python's default tuple sorting handles the timestamp first, then the priority.
        all_events.sort()

        # Process events in the sorted chronological order.
        for timestamp, priority, event_type, data in all_events:
            if event_type == "OFFLINE":
                user_id = data
                # Set the user's status to offline.
                is_online[user_id] = False
            elif event_type == "ONLINE_AGAIN":
                user_id = data
                # Set the user's status to online.
                is_online[user_id] = True
            elif event_type == "MESSAGE":
                mention_string = data
                if mention_string == "ALL":
                    # "ALL" mentions everyone, regardless of online status.
                    for i in range(numberOfUsers):
                        mentions[i] += 1
                elif mention_string == "HERE":
                    # "HERE" mentions only users who are currently online.
                    for i in range(numberOfUsers):
                        if is_online[i]:
                            mentions[i] += 1
                else: # mention_string is in "id..." format (e.g., "id0", "id1 id3 id0")
                    # The string contains one or more user IDs separated by whitespace.
                    mentioned_ids_str = mention_string.split()
                    for id_str in mentioned_ids_str:
                        # Extract the integer user ID from the string (e.g., "id0" -> 0).
                        user_id = int(id_str[2:])
                        # Mention these specific users. The problem states that id<number>
                        # can mention even offline users, so we increment mention count
                        # without checking is_online status.
                        mentions[user_id] += 1

        # Return the final accumulated mention counts for each user.
        return mentions