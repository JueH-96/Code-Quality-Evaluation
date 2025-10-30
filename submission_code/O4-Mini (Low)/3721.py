from typing import List
import heapq

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        # mentions count per user
        mentions = [0] * numberOfUsers
        # online status per user, all start online
        online = [True] * numberOfUsers
        # min‐heap of (reonline_time, user_id)
        reheap = []
        
        # Prepare and sort input events.
        # We'll represent each input event as (time, kind, data)
        # kind: 1 = OFFLINE, 2 = MESSAGE
        ev_list = []
        for ev in events:
            typ, ts_str, data = ev
            t = int(ts_str)
            if typ == "OFFLINE":
                ev_list.append((t, 1, int(data)))   # data is user id
            else:  # "MESSAGE"
                ev_list.append((t, 2, data))        # data is mention string
        
        # Sort by (time, kind) so OFFLINE (1) comes before MESSAGE (2) at same timestamp
        ev_list.sort(key=lambda x: (x[0], x[1]))
        
        # Process events in order
        for t, kind, data in ev_list:
            # First, process any re-online events up to current time
            while reheap and reheap[0][0] <= t:
                re_t, uid = heapq.heappop(reheap)
                online[uid] = True
            
            if kind == 1:
                # OFFLINE event
                uid = data
                # user must be online per spec
                online[uid] = False
                # schedule them to come back at t+60
                heapq.heappush(reheap, (t + 60, uid))
            else:
                # MESSAGE event
                mentions_str = data
                if mentions_str == "ALL":
                    # mention every user
                    for uid in range(numberOfUsers):
                        mentions[uid] += 1
                elif mentions_str == "HERE":
                    # mention all currently online users
                    for uid in range(numberOfUsers):
                        if online[uid]:
                            mentions[uid] += 1
                else:
                    # explicit ids, may have duplicates
                    # tokens like "id3"
                    tokens = mentions_str.split()
                    for tok in tokens:
                        if tok.startswith("id"):
                            uid = int(tok[2:])
                            mentions[uid] += 1
        
        return mentions