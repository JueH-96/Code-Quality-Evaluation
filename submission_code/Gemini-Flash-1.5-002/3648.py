class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        ans = 0
        
        def calculate_fruits(path1, path2, path3):
            visited = set()
            total_fruits = 0
            for r, c in path1:
                if (r,c) not in visited:
                    total_fruits += fruits[r][c]
                    visited.add((r,c))
            for r, c in path2:
                if (r,c) not in visited:
                    total_fruits += fruits[r][c]
                    visited.add((r,c))
            for r, c in path3:
                if (r,c) not in visited:
                    total_fruits += fruits[r][c]
                    visited.add((r,c))
            return total_fruits

        def find_paths(r1, c1, r2, c2, r3, c3, current_paths):
            nonlocal ans
            if r1 == n - 1 and c1 == n - 1 and r2 == n-1 and c2 == n-1 and r3 == n-1 and c3 == n-1:
                ans = max(ans, calculate_fruits(current_paths[0], current_paths[1], current_paths[2]))
                return

            next_paths = []
            
            #Child 1 moves
            moves1 = [(1,1), (1,0), (0,1)]
            for dr1, dc1 in moves1:
                nr1, nc1 = r1 + dr1, c1 + dc1
                if 0 <= nr1 < n and 0 <= nc1 < n:
                    next_paths.append(current_paths[0] + [(nr1, nc1)])

            #Child 2 moves
            moves2 = [(1, -1), (1, 0), (1, 1)]
            for dr2, dc2 in moves2:
                nr2, nc2 = r2 + dr2, c2 + dc2
                if 0 <= nr2 < n and 0 <= nc2 < n:
                    next_paths.append(current_paths[1] + [(nr2, nc2)])

            #Child 3 moves
            moves3 = [(-1, 1), (0, 1), (1, 1)]
            for dr3, dc3 in moves3:
                nr3, nc3 = r3 + dr3, c3 + dc3
                if 0 <= nr3 < n and 0 <= nc3 < n:
                    next_paths.append(current_paths[2] + [(nr3, nc3)])

            
            for i in range(len(next_paths)//3):
                find_paths(next_paths[3*i][-1][0], next_paths[3*i][-1][1], next_paths[3*i+1][-1][0], next_paths[3*i+1][-1][1], next_paths[3*i+2][-1][0], next_paths[3*i+2][-1][1], [next_paths[3*i], next_paths[3*i+1], next_paths[3*i+2]])

        find_paths(0, 0, 0, n - 1, n - 1, 0, [[(0,0)], [(0, n-1)], [(n-1, 0)]])
        return ans