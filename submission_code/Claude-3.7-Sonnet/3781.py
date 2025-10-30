class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        def can_select(d):
            def backtrack(start, count):
                if count == k:
                    return True
                
                for i in range(start, len(points)):
                    valid = True
                    for j in selected:
                        if abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1]) < d:
                            valid = False
                            break
                    
                    if valid:
                        selected.append(i)
                        if backtrack(i + 1, count + 1):
                            return True
                        selected.pop()
                
                return False
            
            selected = []
            return backtrack(0, 0)
        
        # Binary search to find the maximum possible minimum distance
        left, right = 0, 2 * side
        while left < right:
            mid = (left + right + 1) // 2
            if can_select(mid):
                left = mid
            else:
                right = mid - 1
        
        return left