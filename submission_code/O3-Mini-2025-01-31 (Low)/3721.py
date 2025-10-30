from typing import List

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        # Initialize mentions count and offline status (store offline_until timestamp)
        mentions = [0] * numberOfUsers
        # offline_until holds for each user id who is currently offline the time until which they remain offline.
        offline_until = dict()
        
        # First, it's helpful to group events by timestamp
        events_by_time = {}
        for event in events:
            # event[1] is timestamp in string, convert to int
            t = int(event[1])
            if t not in events_by_time:
                events_by_time[t] = []
            events_by_time[t].append(event)
        
        # Process events in order of increasing timestamps
        for t in sorted(events_by_time.keys()):
            # Before processing events at t, update the status of offline users.
            # If any offline_until time has passed, they become online.
            to_remove = []
            for user, until in offline_until.items():
                if until <= t:
                    to_remove.append(user)
            for user in to_remove:
                del offline_until[user]
                
            # Process offline events first (they have precedence per timestamp)
            # We'll separate offline and message events.
            offline_events = []
            message_events = []
            for event in events_by_time[t]:
                if event[0] == "OFFLINE":
                    offline_events.append(event)
                else:
                    message_events.append(event)
            
            # Process OFFLINE events: mark the user offline (for 60 time units)
            for event in offline_events:
                # event structure: ["OFFLINE", timestamp, id]
                user_id = int(event[2])
                # It's guaranteed to be online currently.
                offline_until[user_id] = t + 60
            
            # Now process MESSAGE events
            for event in message_events:
                # event structure: ["MESSAGE", timestamp, mentions_string]
                mention_token = event[2].strip()
                if mention_token == "ALL":
                    # Increments mention count for all users (offline or not)
                    for i in range(numberOfUsers):
                        mentions[i] += 1
                elif mention_token == "HERE":
                    # Increments mention count for users currently online
                    for user in range(numberOfUsers):
                        if user not in offline_until:
                            mentions[user] += 1
                else:
                    # Handle token list like "id1 id0 ..."
                    tokens = mention_token.split()
                    # Each token is of the form "id<number>"
                    for token in tokens:
                        if token.startswith("id"):
                            try:
                                user_id = int(token[2:])
                                if 0 <= user_id < numberOfUsers:
                                    mentions[user_id] += 1
                            except ValueError:
                                # If token does not contain a valid integer (should not happen as per problem statement)
                                pass
            # End of processing events at timestamp t
        
        return mentions