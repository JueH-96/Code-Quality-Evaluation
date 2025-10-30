class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        
        # Create a memoization cache
        memo = {}
        
        def dp(pos1, pos2, pos3):
            # Convert positions to tuples for easier handling
            p1 = (pos1 // n, pos1 % n)
            p2 = (pos2 // n, pos2 % n)
            p3 = (pos3 // n, pos3 % n)
            
            # If any position is out of bounds, return -float('inf')
            if (not (0 <= p1[0] < n and 0 <= p1[1] < n) or
                not (0 <= p2[0] < n and 0 <= p2[1] < n) or
                not (0 <= p3[0] < n and 0 <= p3[1] < n)):
                return -float('inf')
            
            # If we've reached this state before, return memoized result
            state = (pos1, pos2, pos3)
            if state in memo:
                return memo[state]
            
            # Calculate fruits collected in current positions
            current_fruits = 0
            positions = {(p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1])}
            for i, j in positions:
                current_fruits += fruits[i][j]
            
            # If all children reached the target, return current fruits
            if p1 == (n-1, n-1) and p2 == (n-1, n-1) and p3 == (n-1, n-1):
                return current_fruits
            
            # Generate possible next moves for each child
            max_fruits = -float('inf')
            
            # Moves for child 1 (from top-left)
            moves1 = [(p1[0]+1, p1[1]+1), (p1[0]+1, p1[1]), (p1[0], p1[1]+1)]
            # Moves for child 2 (from top-right)
            moves2 = [(p2[0]+1, p2[1]-1), (p2[0]+1, p2[1]), (p2[0]+1, p2[1]+1)]
            # Moves for child 3 (from bottom-left)
            moves3 = [(p3[0]-1, p3[1]+1), (p3[0], p3[1]+1), (p3[0]+1, p3[1]+1)]
            
            # Try all valid combinations of moves
            for m1 in moves1:
                if not (0 <= m1[0] < n and 0 <= m1[1] < n):
                    continue
                for m2 in moves2:
                    if not (0 <= m2[0] < n and 0 <= m2[1] < n):
                        continue
                    for m3 in moves3:
                        if not (0 <= m3[0] < n and 0 <= m3[1] < n):
                            continue
                        next_pos1 = m1[0] * n + m1[1]
                        next_pos2 = m2[0] * n + m2[1]
                        next_pos3 = m3[0] * n + m3[1]
                        max_fruits = max(max_fruits, 
                                      dp(next_pos1, next_pos2, next_pos3))
            
            if max_fruits == -float('inf'):
                max_fruits = 0
                
            memo[state] = current_fruits + max_fruits
            return memo[state]
        
        # Initial positions
        return dp(0, n-1, n*(n-1))