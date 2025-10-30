class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        queue = [x]
        visited = set([x])
        operations = 0
        
        while queue:
            operations += 1
            for _ in range(len(queue)):
                current = queue.pop(0)
                for next_val in [current - 1, current + 1, current // 11 if current % 11 == 0 else float('inf'), current // 5 if current % 5 == 0 else float('inf')]:
                    if next_val == y:
                        return operations
                    if 1 <= next_val <= 10**4 and next_val not in visited:
                        visited.add(next_val)
                        queue.append(next_val)
        return operations