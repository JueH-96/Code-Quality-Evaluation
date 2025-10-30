from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda z: z[1])
        # Pair queries with original indices and sort by query time
        qidx = sorted(enumerate(queries), key=lambda z: z[1])
        
        res = [0] * len(queries)
        cnt = [0] * (n + 1)  # count of requests in current window for each server
        zero = n  # number of servers with zero requests in window
        
        p1 = 0  # remove pointer: logs[p1] are times < (q - x)
        p2 = 0  # add pointer: logs[p2] are times <= q
        
        L = len(logs)
        
        for qi, qtime in qidx:
            # Extend window right end to qtime: add logs with time <= qtime
            while p2 < L and logs[p2][1] <= qtime:
                sid, t = logs[p2]
                cnt[sid] += 1
                if cnt[sid] == 1:
                    zero -= 1
                p2 += 1
            
            # Shrink window left end to qtime - x: remove logs with time < qtime - x
            left_bound = qtime - x
            while p1 < L and logs[p1][1] < left_bound:
                sid, t = logs[p1]
                cnt[sid] -= 1
                if cnt[sid] == 0:
                    zero += 1
                p1 += 1
            
            # Now zero holds number of servers with no requests in [qtime-x, qtime]
            res[qi] = zero
        
        return res