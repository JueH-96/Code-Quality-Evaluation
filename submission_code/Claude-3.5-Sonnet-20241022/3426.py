class Solution:
    def minimumChairs(self, s: str) -> int:
        current_chairs = 0
        min_chairs = 0
        
        for event in s:
            if event == 'E':
                current_chairs += 1
                min_chairs = max(min_chairs, current_chairs)
            else:
                current_chairs -= 1
                
        return min_chairs