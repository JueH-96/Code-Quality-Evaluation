class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        server_counts = [0]*n
        for log in logs:
            server_counts[log[0]-1] += 1
        
        for log in logs:
            server_counts[log[0]-1] -= 1
        
        for i in range(1, n):
            server_counts[i] += server_counts[i-1]
        
        server_counts = [n - c for c in server_counts]
        
        res = []
        for q in queries:
            l = 0
            r = n-1
            while l <= r:
                mid = (l + r) // 2
                if server_counts[mid] >= q:
                    r = mid - 1
                else:
                    l = mid + 1
            res.append(n - l)
        
        return res