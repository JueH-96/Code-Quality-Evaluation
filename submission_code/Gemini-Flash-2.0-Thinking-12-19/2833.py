import bisect

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        server_logs = {}
        for server_id, time in logs:
            if server_id not in server_logs:
                server_logs[server_id] = []
            server_logs[server_id].append(time)
        for server_id in server_logs:
            server_logs[server_id].sort()
        
        result = []
        for q in queries:
            count = 0
            for server_id in range(1, n + 1):
                if server_id not in server_logs:
                    count += 1
                    continue
                times = server_logs[server_id]
                start_time = q - x
                end_time = q
                
                l_idx = bisect.bisect_left(times, start_time)
                found_request = False
                if l_idx < len(times) and times[l_idx] <= end_time:
                    found_request = True
                    
                if not found_request:
                    count += 1
            result.append(count)
        return result