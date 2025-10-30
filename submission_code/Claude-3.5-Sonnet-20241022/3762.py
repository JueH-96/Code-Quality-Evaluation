class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        
        def isAchievable(target):
            # For each index, calculate minimum moves needed to reach target
            moves = [0] * n
            for i in range(n):
                if points[i] < target:
                    return False
                moves[i] = (target + points[i] - 1) // points[i]
            
            # Check if we can achieve this with m moves using two pointers
            left, right = 0, n - 1
            total_moves = 0
            curr_pos = -1
            
            # First move must be to increase index
            total_moves += moves[0]
            curr_pos = 0
            
            while left <= right:
                # If we're at left end
                if curr_pos == left:
                    if left < right:
                        # Move to right if needed
                        moves_needed = max(0, moves[right] - (total_moves - moves[left]))
                        total_moves += moves_needed + (right - left)
                        curr_pos = right
                    left += 1
                # If we're at right end
                else:
                    if left <= right:
                        # Move to left if needed
                        moves_needed = max(0, moves[left] - (total_moves - moves[right]))
                        total_moves += moves_needed + (right - left)
                        curr_pos = left
                    right -= 1
                
                if total_moves > m:
                    return False
            
            return True
        
        # Binary search for the maximum possible minimum value
        left, right = 1, min(points) * (m // 2 + 1)
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            if isAchievable(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result