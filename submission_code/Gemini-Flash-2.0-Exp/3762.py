class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        
        def check(mid):
            moves_needed = float('inf')
            
            # Case 1: Only move right
            temp_moves = 0
            for i in range(n):
                needed = max(0, mid - 0)
                if needed > 0:
                    temp_moves += needed
            if temp_moves <= m:
                moves_needed = min(moves_needed, temp_moves)
            
            # Case 2: Move right then left
            for i in range(n):
                temp_moves = i + 1
                game_score = [0] * n
                for j in range(i + 1):
                    game_score[j] += points[j]
                
                needed = max(0, mid - min(game_score))
                if needed > 0:
                    temp_moves += needed
                
                if temp_moves <= m:
                    moves_needed = min(moves_needed, temp_moves)
            
            # Case 3: Move left then right
            for i in range(n):
                temp_moves = i + 1
                game_score = [0] * n
                for j in range(i + 1):
                    game_score[i-j] += points[i-j]
                
                needed = max(0, mid - min(game_score))
                if needed > 0:
                    temp_moves += needed
                
                if temp_moves <= m:
                    moves_needed = min(moves_needed, temp_moves)
            
            return moves_needed <= m
        
        left, right = 0, max(points) + m
        ans = 0
        
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return ans