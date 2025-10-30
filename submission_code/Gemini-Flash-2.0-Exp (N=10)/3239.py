class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        
        q = [(x, 0)]
        visited = {x}
        
        while q:
            curr_x, steps = q.pop(0)
            
            if curr_x == y:
                return steps
            
            next_moves = []
            
            if curr_x % 11 == 0:
                next_moves.append(curr_x // 11)
            if curr_x % 5 == 0:
                next_moves.append(curr_x // 5)
            next_moves.append(curr_x - 1)
            next_moves.append(curr_x + 1)
            
            for next_x in next_moves:
                if 1 <= next_x <= 20000 and next_x not in visited:
                    q.append((next_x, steps + 1))
                    visited.add(next_x)
        
        return -1