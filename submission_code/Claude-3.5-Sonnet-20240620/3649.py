class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        def can_break_locks(time):
            energy = 0
            factor = 1
            locks = sorted(strength)
            
            for lock in locks:
                while energy < lock and time > 0:
                    energy += factor
                    time -= 1
                
                if energy < lock:
                    return False
                
                energy = 0
                factor += K
            
            return True
        
        left, right = 0, sum(strength)
        
        while left < right:
            mid = (left + right) // 2
            if can_break_locks(mid):
                right = mid
            else:
                left = mid + 1
        
        return left