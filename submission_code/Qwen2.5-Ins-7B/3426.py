class Solution:
    def minimumChairs(self, s: str) -> int:
        max_chairs_needed = 0
        current_chairs = 0
        
        for event in s:
            if event == 'E':
                current_chairs += 1
                max_chairs_needed = max(max_chairs_needed, current_chairs)
            else:
                current_chairs -= 1
        
        return max_chairs_needed