from typing import List
from collections import defaultdict

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # sort logs by time
        logs.sort(key=lambda t: t[1])
        
        # keep the original index of every query so we can restore the order
        indexed_q = sorted([(t, idx) for idx, t in enumerate(queries)])
        
        # frequency array: freq[s] = number of log entries for server s
        freq = [0]*(n+1)          # servers are 1-indexed
        active = 0                # number of servers that currently have freq > 0
        
        res = [0]*len(queries)
        
        add_ptr = 0               # pointer to add logs whose time <= query_time
        rem_ptr = 0               # pointer to remove logs whose time < query_time - x
        L = len(logs)
        
        for q_time, q_idx in indexed_q:
            # 1) add all logs with time <= q_time to the window
            while add_ptr < L and logs[add_ptr][1] <= q_time:
                s = logs[add_ptr][0]
                freq[s] += 1
                if freq[s] == 1:          # server became active
                    active += 1
                add_ptr += 1
            
            # 2) remove logs whose time < q_time - x  (they fall out of the window)
            lower = q_time - x
            while rem_ptr < L and logs[rem_ptr][1] < lower:
                s = logs[rem_ptr][0]
                freq[s] -= 1
                if freq[s] == 0:          # server became inactive
                    active -= 1
                rem_ptr += 1
            
            # 3) servers with zero requests = total servers - active servers
            res[q_idx] = n - active
        
        return res