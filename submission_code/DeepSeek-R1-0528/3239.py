from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        
        bound = 10**5
        visited = [False] * (bound + 1)
        queue = deque([x])
        visited[x] = True
        steps = 0
        
        while queue:
            size = len(queue)
            for _ in range(size):
                u = queue.popleft()
                if u == y:
                    return steps
                neighbors = []
                if u + 1 <= bound:
                    neighbors.append(u + 1)
                if u - 1 >= 0:
                    neighbors.append(u - 1)
                if u % 5 == 0:
                    neighbors.append(u // 5)
                if u % 11 == 0:
                    neighbors.append(u // 11)
                for n in neighbors:
                    if 0 <= n <= bound and not visited[n]:
                        visited[n] = True
                        queue.append(n)
            steps += 1
        
        return -1