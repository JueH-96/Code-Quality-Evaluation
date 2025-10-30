class Solution:
    def minimumChairs(self, s: str) -> int:
        chairs_needed = 0
        current_chairs = 0
        
        for event in s:
            if event == 'E':
                current_chairs += 1
                chairs_needed = max(chairs_needed, current_chairs)
            elif event == 'L':
                current_chairs -= 1
        
        return chairs_needed