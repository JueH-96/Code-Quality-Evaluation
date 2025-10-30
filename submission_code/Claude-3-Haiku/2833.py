class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Create a dictionary to store the requests received by each server
        server_requests = [0] * (n + 1)
        for server_id, time in logs:
            server_requests[server_id] += 1
        
        # Create a list to store the number of servers that received no requests during each query time interval
        result = []
        for query_time in queries:
            no_requests = 0
            for server_id in range(1, n + 1):
                if server_requests[server_id] == 0 and all(query_time - x <= log_time <= query_time for _, log_time in logs if log[0] == server_id):
                    no_requests += 1
            result.append(no_requests)
        
        return result