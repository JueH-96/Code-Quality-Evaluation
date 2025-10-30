class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        logs.sort(key=lambda x: x[1])
        result = []
        for q in queries:
            start_time = q - x
            end_time = q
            servers_with_requests = set()
            left = 0
            right = len(logs) - 1
            while left <= right:
                mid = (left + right) // 2
                if logs[mid][1] >= start_time:
                    right = mid - 1
                else:
                    left = mid + 1
            start_index = left
            left = 0
            right = len(logs) - 1
            while left <= right:
                mid = (left + right) // 2
                if logs[mid][1] <= end_time:
                    left = mid + 1
                else:
                    right = mid - 1
            end_index = right
            for i in range(start_index, end_index + 1):
                servers_with_requests.add(logs[i][0])
            result.append(n - len(servers_with_requests))
        return result