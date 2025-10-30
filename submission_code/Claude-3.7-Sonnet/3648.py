class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        
        # Define the possible moves for each child
        # Child 1: (0,0) -> (n-1,n-1)
        # Child 2: (0,n-1) -> (n-1,n-1)
        # Child 3: (n-1,0) -> (n-1,n-1)
        
        # Create memoization dictionaries for each child
        memo1 = {}
        memo2 = {}
        memo3 = {}
        
        # Define recursive functions to calculate max fruits for each child's path
        def dp1(i, j, moves):
            if moves == 0 and i == n-1 and j == n-1:
                return fruits[i][j]
            if moves == 0 or i >= n or j >= n:
                return float('-inf')
            
            if (i, j, moves) in memo1:
                return memo1[(i, j, moves)]
            
            # Possible moves for child 1: (i+1,j+1), (i+1,j), (i,j+1)
            result = max(
                dp1(i+1, j+1, moves-1),
                dp1(i+1, j, moves-1),
                dp1(i, j+1, moves-1)
            )
            
            memo1[(i, j, moves)] = result + fruits[i][j]
            return memo1[(i, j, moves)]
        
        def dp2(i, j, moves):
            if moves == 0 and i == n-1 and j == n-1:
                return fruits[i][j]
            if moves == 0 or i >= n or j < 0 or j >= n:
                return float('-inf')
            
            if (i, j, moves) in memo2:
                return memo2[(i, j, moves)]
            
            # Possible moves for child 2: (i+1,j-1), (i+1,j), (i+1,j+1)
            result = max(
                dp2(i+1, j-1, moves-1) if j > 0 else float('-inf'),
                dp2(i+1, j, moves-1),
                dp2(i+1, j+1, moves-1) if j+1 < n else float('-inf')
            )
            
            memo2[(i, j, moves)] = result + fruits[i][j]
            return memo2[(i, j, moves)]
        
        def dp3(i, j, moves):
            if moves == 0 and i == n-1 and j == n-1:
                return fruits[i][j]
            if moves == 0 or i < 0 or i >= n or j >= n:
                return float('-inf')
            
            if (i, j, moves) in memo3:
                return memo3[(i, j, moves)]
            
            # Possible moves for child 3: (i-1,j+1), (i,j+1), (i+1,j+1)
            result = max(
                dp3(i-1, j+1, moves-1) if i > 0 else float('-inf'),
                dp3(i, j+1, moves-1),
                dp3(i+1, j+1, moves-1) if i+1 < n else float('-inf')
            )
            
            memo3[(i, j, moves)] = result + fruits[i][j]
            return memo3[(i, j, moves)]
        
        # Recover the paths for each child
        def get_path(dp_func, memo, start_i, start_j):
            path = [(start_i, start_j)]
            i, j = start_i, start_j
            moves = n-1
            
            while moves > 0:
                best_move = None
                best_val = float('-inf')
                
                if dp_func == dp1:
                    possible_moves = [(i+1, j+1), (i+1, j), (i, j+1)]
                elif dp_func == dp2:
                    possible_moves = [(i+1, j-1), (i+1, j), (i+1, j+1)]
                else:  # dp3
                    possible_moves = [(i-1, j+1), (i, j+1), (i+1, j+1)]
                
                for ni, nj in possible_moves:
                    if 0 <= ni < n and 0 <= nj < n:
                        val = memo.get((ni, nj, moves-1), float('-inf'))
                        if val > best_val:
                            best_val = val
                            best_move = (ni, nj)
                
                if best_move:
                    i, j = best_move
                    path.append((i, j))
                moves -= 1
            
            return path
        
        # Calculate max fruits for each child's path
        dp1(0, 0, n-1)
        dp2(0, n-1, n-1)
        dp3(n-1, 0, n-1)
        
        # Get optimal paths
        path1 = get_path(dp1, memo1, 0, 0)
        path2 = get_path(dp2, memo2, 0, n-1)
        path3 = get_path(dp3, memo3, n-1, 0)
        
        # Calculate total fruits collected, considering the rule that
        # fruits can only be collected once
        visited = set()
        total_fruits = 0
        
        for pos in path1 + path2 + path3:
            if pos not in visited:
                total_fruits += fruits[pos[0]][pos[1]]
                visited.add(pos)
        
        return total_fruits