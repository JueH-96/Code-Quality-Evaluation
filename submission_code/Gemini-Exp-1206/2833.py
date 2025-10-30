class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        ans = []
        for q in queries:
            start = q - x
            end = q
            
            servers_with_requests = set()
            for log in logs:
                server_id, time = log
                if start <= time <= end:
                    servers_with_requests.add(server_id)
            
            ans.append(n - len(servers_with_requests))
        
        return ans