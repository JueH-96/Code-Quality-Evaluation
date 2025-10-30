class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        s = {changeIndices[i] - 1 for i in range(len(changeIndices))}
        if len(s) != n:
            return -1
        
        topog_order = []
        indegree = [0] * n
        visited = set()
        
        adj_list = {i: set() for i in range(n)}
        for i, ind in enumerate(changeIndices[1:], 1):
            if nums[ind - 1] > 0 and ind - 1 not in visited:
                indegree[ind - 1] += 1
                if ind - 2 >= 0:
                    adj_list[ind - 2].add(ind - 1)
            else:
                nums[ind - 1] = 0
                visited.add(ind - 1)
        
        queue = [i for i in range(n) if indegree[i] == 0]
        
        while queue:
            node = queue.pop(0)
            topog_order.append(node)
            for neighbor in adj_list[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0 and neighbor not in visited:
                    queue.append(neighbor)
        
        if len(topog_order) != n:
            return -1
        
        timestamps = []
        for ind in topog_order:
            if nums[ind] != 0:
                timestamps.append((nums[ind], ind))
        
        timestamps.sort()
        
        for i in range(len(timestamps)):
            nums[timestamps[i][1]] = 0
            visited.add(timestamps[i][1])
        
        for i, node in enumerate(topog_order):
            if node not in visited:
                timestamps.insert(i, (0, node))
        
        answer = 0
        
        for i, node in enumerate(topog_order):
            answer += timestamps[i][0] + 1 # +1 to convert to 1 based indexing
            if answer > len(changeIndices):
                return -1
        
        return answer