class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        logs.sort(key=lambda item: item[1])
        results = []
        for q_time in queries:
            start_time = q_time - x
            end_time = q_time
            active_servers = set()
            for log in logs:
                server_id, log_time = log
                if start_time <= log_time <= end_time:
                    active_servers.add(server_id)
            inactive_count = n - len(active_servers)
            results.append(inactive_count)
        return results