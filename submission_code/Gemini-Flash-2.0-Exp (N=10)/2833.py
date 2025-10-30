class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        ans = []
        for q in queries:
            count = 0
            servers_with_requests = set()
            for log in logs:
                if q - x <= log[1] <= q:
                    servers_with_requests.add(log[0])
            count = n - len(servers_with_requests)
            ans.append(count)
        return ans