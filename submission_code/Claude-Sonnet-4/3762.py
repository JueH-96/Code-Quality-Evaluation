class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        
        def minMovesNeeded(target):
            # Calculate required visits for each position
            required = [(target + p - 1) // p for p in points]
            
            min_moves = float('inf')
            
            # Try all possible ranges [l, r] that we might visit
            for l in range(n):
                for r in range(l, n):
                    # Check if visiting range [l, r] is sufficient
                    sufficient = True
                    for i in range(n):
                        if required[i] > 0 and (i < l or i > r):
                            sufficient = False
                            break
                    
                    if not sufficient:
                        continue
                    
                    # Calculate minimum moves for this range
                    total_required = sum(required[l:r+1])
                    range_size = r - l + 1
                    
                    # We need to enter the array (1 move) and reach position l (l more moves)
                    # Then we need total_required visits in range [l, r]
                    # Optimal strategy: bounce between l and r
                    moves_to_enter = l + 1
                    
                    if total_required <= range_size:
                        # We can satisfy requirements with single traversal
                        moves = moves_to_enter + (r - l)
                    else:
                        # We need extra visits
                        extra_visits = total_required - range_size
                        moves = moves_to_enter + (r - l) + extra_visits * 2
                    
                    min_moves = min(min_moves, moves)
            
            return min_moves
        
        # Binary search on answer
        left, right = 0, min(points) * m
        
        while left < right:
            mid = (left + right + 1) // 2
            if minMovesNeeded(mid) <= m:
                left = mid
            else:
                right = mid - 1
        
        return left