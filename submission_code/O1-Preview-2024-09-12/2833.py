class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        from collections import defaultdict
        events = []
        for server_id, time in logs:
            events.append((time, 0, server_id))               # start event
            events.append((time + x + 1, 2, server_id))       # end event

        indexed_queries = sorted([(q, idx) for idx, q in enumerate(queries)])
        for q, idx in indexed_queries:
            events.append((q + 1, 1, idx))                    # query event

        # Sort events: time, event_type (0=start, 1=query, 2=end)
        # We process 'start' events first, then 'query', then 'end' to ensure correct processing
        events.sort()
        results = [0] * len(queries)
        count_active_servers = 0
        active_servers = set()

        for time, event_type, data in events:
            if event_type == 0:  # start event
                if data not in active_servers:
                    active_servers.add(data)
                    count_active_servers += 1
            elif event_type == 2:  # end event
                if data in active_servers:
                    active_servers.remove(data)
                    count_active_servers -= 1
            else:  # query event
                idx = data
                results[idx] = n - count_active_servers

        return results