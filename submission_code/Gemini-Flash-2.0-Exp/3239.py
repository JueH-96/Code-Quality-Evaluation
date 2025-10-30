class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x >= y:
            q = [(x, 0)]
            visited = {x}
            while q:
                curr, steps = q.pop(0)
                if curr == y:
                    return steps
                
                if curr % 11 == 0 and curr // 11 > 0 and curr // 11 <= 10000 and curr // 11 not in visited:
                    q.append((curr // 11, steps + 1))
                    visited.add(curr // 11)
                if curr % 5 == 0 and curr // 5 > 0 and curr // 5 <= 10000 and curr // 5 not in visited:
                    q.append((curr // 5, steps + 1))
                    visited.add(curr // 5)
                if curr - 1 > 0 and curr - 1 <= 10000 and curr - 1 not in visited:
                    q.append((curr - 1, steps + 1))
                    visited.add(curr - 1)
                if curr + 1 <= 10000 and curr + 1 not in visited:
                    q.append((curr + 1, steps + 1))
                    visited.add(curr + 1)
        else:
            return y - x