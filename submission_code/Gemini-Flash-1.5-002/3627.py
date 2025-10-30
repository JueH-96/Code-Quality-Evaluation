class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        
        queue = [(0, 0, 0)] # (row, col, time)
        visited = set()
        visited.add((0,0))

        while queue:
            row, col, time = queue.pop(0)

            if row == n - 1 and col == m - 1:
                return time

            # Explore adjacent rooms
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < n and 0 <= new_col < m:
                    arrival_time = max(time + 1, moveTime[new_row][new_col])
                    if (new_row, new_col) not in visited:
                        visited.add((new_row, new_col))
                        queue.append((new_row, new_col, arrival_time))
                        
        return -1 #Should not reach here given constraints