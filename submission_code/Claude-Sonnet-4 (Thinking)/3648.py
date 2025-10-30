class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        memo = {}
        
        def dp(move_num, r1, c1, r2, c2, r3, c3):
            # Base case: after n-1 moves, check if all children reached destination
            if move_num == n - 1:
                if r1 == n-1 and c1 == n-1 and r2 == n-1 and c2 == n-1 and r3 == n-1 and c3 == n-1:
                    return 0
                else:
                    return -1  # Invalid state
            
            state = (move_num, r1, c1, r2, c2, r3, c3)
            if state in memo:
                return memo[state]
            
            max_fruits = -1
            
            # Possible moves for each child
            moves1 = [(r1+1, c1+1), (r1+1, c1), (r1, c1+1)]  # down-right, down, right
            moves2 = [(r2+1, c2-1), (r2+1, c2), (r2+1, c2+1)]  # down-left, down, down-right
            moves3 = [(r3-1, c3+1), (r3, c3+1), (r3+1, c3+1)]  # up-right, right, down-right
            
            # Filter moves within bounds
            moves1 = [(r, c) for r, c in moves1 if 0 <= r < n and 0 <= c < n]
            moves2 = [(r, c) for r, c in moves2 if 0 <= r < n and 0 <= c < n]
            moves3 = [(r, c) for r, c in moves3 if 0 <= r < n and 0 <= c < n]
            
            # Try all combinations of moves
            for nr1, nc1 in moves1:
                for nr2, nc2 in moves2:
                    for nr3, nc3 in moves3:
                        # Calculate fruits collected this turn
                        collected = 0
                        visited = set()
                        
                        for nr, nc in [(nr1, nc1), (nr2, nc2), (nr3, nc3)]:
                            if (nr, nc) not in visited:
                                collected += fruits[nr][nc]
                                visited.add((nr, nc))
                        
                        # Recurse for remaining moves
                        future = dp(move_num + 1, nr1, nc1, nr2, nc2, nr3, nc3)
                        if future != -1:
                            if max_fruits == -1:
                                max_fruits = collected + future
                            else:
                                max_fruits = max(max_fruits, collected + future)
            
            memo[state] = max_fruits
            return max_fruits
        
        # Collect fruits from initial positions
        initial_fruits = 0
        visited_initial = set()
        
        for r, c in [(0, 0), (0, n-1), (n-1, 0)]:
            if (r, c) not in visited_initial:
                initial_fruits += fruits[r][c]
                visited_initial.add((r, c))
        
        # Start DP from initial positions
        result = dp(0, 0, 0, 0, n-1, n-1, 0)
        
        return initial_fruits + result if result != -1 else 0