import heapq

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        n = len(usageLimits)
        low = 0
        high = n # Let's try n as an upper bound for the number of groups
        max_groups = 0
        while low <= high:
            k = (low + high) // 2
            if k == 0:
                max_groups = max(max_groups, 0)
                low = 1
                continue
            current_usage_limits = list(usageLimits)
            possible = True
            for group_size in range(1, k + 1):
                usage_limit_indices = []
                for i in range(n):
                    usage_limit_indices.append((-current_usage_limits[i], i))
                heapq.heapify(usage_limit_indices)
                selected_indices = []
                for _ in range(group_size):
                    if not usage_limit_indices:
                        possible = False
                        break
                    limit, index = heapq.heappop(usage_limit_indices)
                    if -limit <= 0:
                        possible = False
                        break
                    selected_indices.append(index)
                if not possible:
                    break
                for index in selected_indices:
                    current_usage_limits[index] -= 1
            if possible:
                max_groups = max(max_groups, k)
                low = k + 1
            else:
                high = k - 1
        return max_groups