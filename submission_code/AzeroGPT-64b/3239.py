class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        q = deque([(x, 0)])
        visited = set([x])
        while q:
            x, step = q.popleft()
            if x == y:
                return step
            if x % 11 == 0 and x // 11 not in visited:
                visited.add(x // 11)
                q.append((x // 11, step + 1))
            if x % 5 == 0 and x // 5 not in visited:
                visited.add(x // 5)
                q.append((x // 5, step + 1))
            if x - 1 not in visited and x - 1 > 0:
                visited.add(x - 1)
                q.append((x - 1, step + 1))
            if x + 1 not in visited and x + 1 < 10000:
                visited.add(x + 1)
                q.append((x + 1, step + 1))
        
        return -1