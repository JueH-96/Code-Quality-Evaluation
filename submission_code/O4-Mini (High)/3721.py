from typing import List

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        # We'll build a list of all "status change" and "message" events,
        # tagging them with a priority so that at the same timestamp:
        #   0 = scheduled "come back online"
        #   1 = explicit "go offline"
        #   2 = MESSAGE
        processed_events = []
        
        for evt in events:
            typ = evt[0]
            t = int(evt[1])
            if typ == "OFFLINE":
                uid = int(evt[2])
                # Schedule the automatic "come back online" at t+60
                processed_events.append((t + 60, 0, uid))
                # Process the explicit offline right at t
                processed_events.append((t, 1, uid))
            else:  # "MESSAGE"
                mentions_str = evt[2]
                processed_events.append((t, 2, mentions_str))
        
        # Sort by (timestamp, priority)
        processed_events.sort(key=lambda x: (x[0], x[1]))
        
        online = [True] * numberOfUsers
        mentions = [0] * numberOfUsers
        
        for _, ev_type, data in processed_events:
            if ev_type == 0:
                # scheduled come-back
                online[data] = True
            elif ev_type == 1:
                # go offline
                online[data] = False
            else:
                # MESSAGE event
                s = data
                if s == "ALL":
                    # mention every user, online or offline
                    for i in range(numberOfUsers):
                        mentions[i] += 1
                elif s == "HERE":
                    # mention only those currently online
                    for i in range(numberOfUsers):
                        if online[i]:
                            mentions[i] += 1
                else:
                    # a list of id<number> tokens, possibly with duplicates
                    for token in s.split():
                        # token is like 'id3'
                        uid = int(token[2:])
                        mentions[uid] += 1
        
        return mentions