from collections import defaultdict
from typing import List

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        online_status = [True] * numberOfUsers # True if online, False if offline
        offline_until = {} # Dictionary: {user_id: timestamp_becomes_online_again}

        # Group events by timestamp and type for processing order
        # sort_key: 0 for OFFLINE, 1 for MESSAGE
        # This ensures OFFLINE events at a timestamp are processed before MESSAGE events
        timestamp_groups = defaultdict(list)
        for event in events:
            timestamp = int(event[1])
            event_type = event[0]
            if event_type == "OFFLINE":
                # The user goes offline at 'timestamp' and comes back online at 'timestamp + 60'.
                # The status change happens *at* 'timestamp'.
                # We schedule their return for 'timestamp + 60'.
                timestamp_groups[timestamp].append((0, event))
            elif event_type == "MESSAGE":
                 # Message events are processed after status changes at the same timestamp.
                timestamp_groups[timestamp].append((1, event))

        # Collect all relevant time points: event timestamps and offline return times.
        all_relevant_times = set(timestamp_groups.keys()) # Start with event timestamps
        # Add return times for users who might go offline
        for event in events:
            if event[0] == "OFFLINE":
                offline_time = int(event[1])
                return_online_time = offline_time + 60
                all_relevant_times.add(return_online_time)

        # Sort all unique relevant time points
        sorted_relevant_times = sorted(list(all_relevant_times))

        # Process time step by time step using all relevant times
        for current_timestamp in sorted_relevant_times:
            # Phase 1: Process users coming online whose offline period ends exactly at or before this timestamp.
            # This must happen before any events *at* this timestamp are processed.
            # Iterate over a copy of keys because we might delete from offline_until
            users_to_bring_online = [
                user_id for user_id, return_time in list(offline_until.items())
                if return_time <= current_timestamp
            ]
            for user_id in users_to_bring_online:
                online_status[user_id] = True
                del offline_until[user_id]

            # Phase 2: Process events that occur exactly at the current timestamp.
            # Sort events at this timestamp: OFFLINE (0) before MESSAGE (1)
            # If there are no events at this timestamp, timestamp_groups.get(current_timestamp, []) will be empty, and this loop is skipped.
            events_at_this_timestamp = sorted(timestamp_groups.get(current_timestamp, []))

            for sort_key, event in events_at_this_timestamp:
                event_type = event[0]
                # event_time_str = event[1] # Not needed, we have current_timestamp
                event_data = event[2]

                if event_type == "OFFLINE":
                    user_id = int(event_data)
                    # Problem guarantees user is online when OFFLINE event occurs.
                    online_status[user_id] = False
                    offline_until[user_id] = current_timestamp + 60

                elif event_type == "MESSAGE":
                    if event_data == "ALL":
                        for user_id in range(numberOfUsers):
                            mentions[user_id] += 1
                    elif event_data == "HERE":
                        for user_id in range(numberOfUsers):
                            if online_status[user_id]: # Check status *after* handling users coming online and OFFLINE events at this timestamp
                                mentions[user_id] += 1
                    else: # id<number> id<number> ... format
                        mention_tokens = event_data.split()
                        for token in mention_tokens:
                            # Extract number after "id"
                            user_id_str = token[2:]
                            user_id = int(user_id_str)
                            # Problem states "This can mention even the offline users." for this format
                            mentions[user_id] += 1

        return mentions