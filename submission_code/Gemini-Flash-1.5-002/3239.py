class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        q = [(x, 0)]
        visited = {x}
        while q:
            curr_x, steps = q.pop(0)
            if curr_x == y:
                return steps
            
            next_states = []
            if curr_x % 11 == 0:
                next_states.append(curr_x // 11)
            if curr_x % 5 == 0:
                next_states.append(curr_x // 5)
            next_states.extend([curr_x -1, curr_x + 1])

            for next_x in next_states:
                if 1 <= next_x <= 10000 and next_x not in visited:
                    visited.add(next_x)
                    q.append((next_x, steps + 1))
        return -1 #Should not reach here given constraints