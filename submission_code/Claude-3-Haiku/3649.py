class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        left, right = 1, sum(strength)
        
        while left <= right:
            mid = (left + right) // 2
            energy, factor = 0, 1
            
            for i in range(n):
                if energy < strength[i]:
                    energy = 0
                    factor += K
                energy += factor
            
            if energy >= sum(strength):
                right = mid - 1
            else:
                left = mid + 1
        
        return left