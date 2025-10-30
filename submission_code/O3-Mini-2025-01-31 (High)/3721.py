from typing import List

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        # We'll simulate the timeline by merging the input events with the automatic ONLINE events.
        # Priority is given so that all status change events (ONLINE and OFFLINE) are processed before MESSAGE events.
        # Also, among status events at the same timestamp, we process ONLINE events first (so that if an offline event 
        # also occurs at the same timestamp, the user is indeed online when the offline event triggers, as required).
        
        # We'll assign priorities as follows:
        #   ONLINE event: priority 0
        #   OFFLINE event: priority 1
        #   MESSAGE event: priority 2
        scheduled_events = []
        for event in events:
            event_type = event[0]
            timestamp = int(event[1])
            if event_type == "OFFLINE":
                # The third element is the id (string), convert to int.
                user_id = int(event[2])
                # Append the OFFLINE event with priority 1.
                scheduled_events.append((timestamp, 1, "OFFLINE", user_id))
                # Schedule the automatic ONLINE event for time = timestamp + 60 with priority 0.
                scheduled_events.append((timestamp + 60, 0, "ONLINE", user_id))
            elif event_type == "MESSAGE":
                message_str = event[2]
                # Append the MESSAGE event with priority 2.
                scheduled_events.append((timestamp, 2, "MESSAGE", message_str))
        
        # Sort events by (timestamp, priority)
        scheduled_events.sort(key=lambda ev: (ev[0], ev[1]))
        
        # All users start as online.
        online = [True] * numberOfUsers
        mentions = [0] * numberOfUsers
        
        # Process the scheduled events in sorted order.
        for ev in scheduled_events:
            time, _, ev_type, data = ev
            if ev_type == "ONLINE":
                # Reactivation event: mark the user as online.
                user_id = data
                online[user_id] = True
            elif ev_type == "OFFLINE":
                # Offline event: mark the user as offline.
                user_id = data
                online[user_id] = False
            elif ev_type == "MESSAGE":
                # Process the message event.
                msg = data
                tokens = msg.split()  # Tokens can be "ALL", "HERE", or "id<number>".
                for token in tokens:
                    if token == "ALL":
                        # "ALL" mentions every user regardless of their online/offline status.
                        for i in range(numberOfUsers):
                            mentions[i] += 1
                    elif token == "HERE":
                        # "HERE" mentions only the users who are currently online.
                        for i in range(numberOfUsers):
                            if online[i]:
                                mentions[i] += 1
                    elif token.startswith("id"):
                        # Token is of the form "id<number>".
                        try:
                            user_id = int(token[2:])
                        except:
                            continue
                        if 0 <= user_id < numberOfUsers:
                            mentions[user_id] += 1
        return mentions