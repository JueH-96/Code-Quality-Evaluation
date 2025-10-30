class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        logs.sort()
        server_requests = [0] * (n + 1)
        intervals = []
        for log in logs:
            intervals.append((log[1], log[0]))
        intervals.sort()
        l = 0
        r = 0
        res = [0] * len(queries)
        query_index = 0
        while query_index < len(queries):
            while l < len(intervals) and intervals[l][0] <= queries[query_index]:
                server_requests[intervals[l][1]] += 1
                l += 1
            while r < len(intervals) and intervals[r][0] <= queries[query_index] + x:
                server_requests[intervals[r][1]] += 1
                r += 1
            for i in range(1, n + 1):
                if server_requests[i] == 0:
                    res[query_index] += 1
            for i in range(1, n + 1):
                server_requests[i] = 0
            query_index += 1
        return res