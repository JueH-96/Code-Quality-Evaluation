class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        
        def canAchieve(target):
            # Calculate required visits for each index
            visits = []
            for i in range(n):
                needed = (target + points[i] - 1) // points[i]  # ceil(target / points[i])
                visits.append(needed)
            
            # Calculate minimum moves needed
            # We need to visit each index visits[i] times
            # Starting from -1, we need to find optimal path
            
            # Total visits needed
            total_visits = sum(visits)
            
            # Minimum moves is at least total_visits
            if total_visits > m:
                return False
            
            # We need to check if we can visit all positions with required frequencies
            # within m moves starting from position -1
            
            # The optimal strategy is to minimize back-and-forth movements
            # We can use a greedy approach: visit positions in order while fulfilling requirements
            
            # Calculate minimum moves needed
            min_moves = 0
            
            # First, we need to reach position 0 (1 move)
            min_moves += 1
            
            # Then we need to visit all positions with required frequencies
            # The minimum way is to go through all positions collecting as we go
            
            # We need at least (n-1) moves to reach the last position
            # Plus additional moves for extra visits
            
            # Total moves = initial move + moves to traverse + extra visits
            # Extra visits = total_visits - n (since we visit each position at least once while traversing)
            
            if total_visits <= n:
                # We can visit each position at most once
                min_moves = n  # 1 to reach 0, then n-1 to reach end
            else:
                # We need multiple visits
                # Minimum moves = n + (total_visits - n) = total_visits
                min_moves = total_visits
            
            return min_moves <= m
        
        # Binary search on the answer
        left, right = 1, max(points) * m
        
        while left < right:
            mid = (left + right + 1) // 2
            if canAchieve(mid):
                left = mid
            else:
                right = mid - 1
        
        return left