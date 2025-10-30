class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        events = [(t + x, 1, sid) for sid, t in logs] + list(zip(queries, range(len(queries)), range(n, 2 * n)))
        events.sort()
        
        ans = [0] * len(queries)
        active = [False] * n
        s = sorted(logs, key=lambda k: k[1])
        j = count = 0
        
        for _, qid, sid in events:
            if sid < n:
                if not active[sid]:
                    count += 1
                    active[sid] = True
            else:
                while j < len(logs) and s[j][1] <= events[qid][0]:
                    if not active[s[j][0]]:
                        count += 1
                        active[s[j][0]] = True
                    j += 1
                ans[qid - len(queries)] = n - count
                active[sid - n] = False
                count -= 1
        return ans