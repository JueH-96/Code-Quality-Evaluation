class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        from collections import deque
        
        n = len(nums)
        if n == 1:
            return 0
        
        # Graph representation: adjacency list
        adj = [[] for _ in range(n)]
        
        # Build the graph based on the jump condition
        for i in range(n):
            for j in range(i + 1, n):
                if abs(nums[j] - nums[i]) <= target:
                    adj[i].append(j)
        
        # BFS to find the longest path from 0 to n-1
        queue = deque([(0, 0)])  # (current_index, jump_count)
        visited = [-1] * n  # to store the max jumps to reach each index
        visited[0] = 0
        
        max_jumps = -1
        
        while queue:
            current, jumps = queue.popleft()
            
            for neighbor in adj[current]:
                if jumps + 1 > visited[neighbor]:
                    visited[neighbor] = jumps + 1
                    queue.append((neighbor, jumps + 1))
                    if neighbor == n - 1:
                        max_jumps = max(max_jumps, jumps + 1)
        
        return max_jumps