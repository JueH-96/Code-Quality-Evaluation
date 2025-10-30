from typing import List
import heapq

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        # Parse and sort events by timestamp, with OFFLINE before MESSAGE at same timestamp
        # Each event is [type, timestamp_str, payload]
        events_sorted = sorted(events, key=lambda e: (int(e[1]), 0 if e[0] == "OFFLINE" else 1))
        
        mentions = [0] * numberOfUsers
        online = [True] * numberOfUsers
        
        # Min-heap of (reactivation_time, user_id)
        reactivation_heap = []
        
        def process_reactivations(up_to_time: int):
            # Bring back online any users whose reactivation_time <= up_to_time
            while reactivation_heap and reactivation_heap[0][0] <= up_to_time:
                react_time, uid = heapq.heappop(reactivation_heap)
                online[uid] = True
        
        for evt in events_sorted:
            typ, ts_str, payload = evt
            t = int(ts_str)
            # First, process any scheduled reactivations up to current time
            process_reactivations(t)
            
            if typ == "OFFLINE":
                uid = int(payload)
                # user goes offline now
                online[uid] = False
                # schedule reactivation at t + 60
                heapq.heappush(reactivation_heap, (t + 60, uid))
            
            else:  # MESSAGE
                # payload is mentions_string
                if payload == "ALL":
                    # mention every user, regardless of online
                    for uid in range(numberOfUsers):
                        mentions[uid] += 1
                elif payload == "HERE":
                    # mention only those currently online
                    for uid in range(numberOfUsers):
                        if online[uid]:
                            mentions[uid] += 1
                else:
                    # parse id<number> tokens, count duplicates
                    tokens = payload.split()
                    for tok in tokens:
                        # tok is like "id3"
                        if tok.startswith("id"):
                            uid = int(tok[2:])
                            mentions[uid] += 1
        
        return mentions