class Solution:
    def minimumChairs(self, s: str) -> int:
        current_people = 0
        max_chairs_needed = 0
        
        for event in s:
            if event == 'E':
                current_people += 1
            elif event == 'L':
                current_people -= 1
            
            max_chairs_needed = max(max_chairs_needed, current_people)
        
        return max_chairs_needed