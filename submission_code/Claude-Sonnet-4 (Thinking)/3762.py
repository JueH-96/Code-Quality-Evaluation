import math

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        def can_achieve(target):
            n = len(points)
            need = [math.ceil(target / points[i]) for i in range(n)]
            
            pos = 0
            moves = 1
            need[0] -= 1
            
            while sum(need) > 0 and moves < m:
                # Find the adjacent position with the highest need
                best_pos = pos
                best_need = -1
                
                if pos > 0 and need[pos - 1] > best_need:
                    best_need = need[pos - 1]
                    best_pos = pos - 1
                
                if pos < n - 1 and need[pos + 1] > best_need:
                    best_need = need[pos + 1]
                    best_pos = pos + 1
                
                # If no adjacent position has a positive need, find the closest position with a positive need
                if best_need <= 0:
                    closest_pos = -1
                    min_dist = float('inf')
                    for i in range(n):
                        if need[i] > 0:
                            dist = abs(i - pos)
                            if dist < min_dist:
                                min_dist = dist
                                closest_pos = i
                    
                    if closest_pos == -1:
                        break  # All needs are satisfied
                    
                    # Move towards the closest position
                    if closest_pos > pos:
                        best_pos = pos + 1
                    else:
                        best_pos = pos - 1
                
                pos = best_pos
                moves += 1
                need[pos] -= 1
            
            return sum(need) <= 0
        
        # Binary search on the answer
        upper_bound = int(m / sum(1 / p for p in points))
        left, right = 0, upper_bound
        while left < right:
            mid = (left + right + 1) // 2
            if can_achieve(mid):
                left = mid
            else:
                right = mid - 1
        
        return left