class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        logs.sort(key=lambda x: x[1])
        ans = []
        for q in queries:
            count = 0
            servers = set(range(1, n + 1))
            l, r = 0, len(logs) - 1
            while l <= r:
                mid = (l + r) // 2
                if logs[mid][1] <= q - x:
                    l = mid + 1
                else:
                    r = mid - 1
            
            active_servers = set()
            for i in range(l, len(logs)):
                if logs[i][1] <= q:
                    active_servers.add(logs[i][0])
                else:
                    break

            count = n - len(active_servers)
            ans.append(count)
        return ans