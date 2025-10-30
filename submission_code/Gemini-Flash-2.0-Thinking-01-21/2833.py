class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        server_logs = {}
        for log in logs:
            server_id, time = log
            if server_id not in server_logs:
                server_logs[server_id] = []
            server_logs[server_id].append(time)
        
        result = []
        for q in queries:
            start_time = q - x
            no_request_servers = 0
            for server_id in range(1, n + 1):
                received_request = False
                if server_id in server_logs:
                    for time in server_logs[server_id]:
                        if start_time <= time <= q:
                            received_request = True
                            break
                if not received_request:
                    no_request_servers += 1
            result.append(no_request_servers)
        return result