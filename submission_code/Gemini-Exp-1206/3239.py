class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        q = [(x, 0)]
        visited = {x}
        while q:
            curr_x, steps = q.pop(0)
            if curr_x == y:
                return steps
            if curr_x - 1 >= 0 and curr_x - 1 not in visited:
                q.append((curr_x - 1, steps + 1))
                visited.add(curr_x - 1)
            if curr_x + 1 <= 100000 and curr_x + 1 not in visited:
                q.append((curr_x + 1, steps + 1))
                visited.add(curr_x + 1)
            if curr_x % 11 == 0 and curr_x // 11 not in visited:
                q.append((curr_x // 11, steps + 1))
                visited.add(curr_x // 11)
            if curr_x % 5 == 0 and curr_x // 5 not in visited:
                q.append((curr_x // 5, steps + 1))
                visited.add(curr_x // 5)
            q.sort(key=lambda x: abs(x[0] - y) + x[1])
        return -1