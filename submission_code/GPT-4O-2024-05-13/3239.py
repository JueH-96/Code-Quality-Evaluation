class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque
        
        # BFS approach to find the minimum number of operations
        queue = deque([(x, 0)])  # (current value, number of operations)
        visited = set()
        
        while queue:
            current, operations = queue.popleft()
            
            if current == y:
                return operations
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Generate possible next states
            if current % 11 == 0:
                queue.append((current // 11, operations + 1))
            if current % 5 == 0:
                queue.append((current // 5, operations + 1))
            queue.append((current - 1, operations + 1))
            queue.append((current + 1, operations + 1))
        
        return -1  # This should never be reached

# Example usage:
# sol = Solution()
# print(sol.minimumOperationsToMakeEqual(26, 1))  # Output: 3
# print(sol.minimumOperationsToMakeEqual(54, 2))  # Output: 4
# print(sol.minimumOperationsToMakeEqual(25, 30))  # Output: 5