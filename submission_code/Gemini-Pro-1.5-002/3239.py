class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        q = [(x, 0)]
        visited = {x}
        while q:
            curr, steps = q.pop(0)
            if curr == y:
                return steps
            nexts = []
            if curr % 11 == 0:
                nexts.append(curr // 11)
            if curr % 5 == 0:
                nexts.append(curr // 5)
            nexts.append(curr + 1)
            nexts.append(curr - 1)

            for next_val in nexts:
                if 1 <= next_val <= 10000 and next_val not in visited:
                    q.append((next_val, steps + 1))
                    visited.add(next_val)
        return -1