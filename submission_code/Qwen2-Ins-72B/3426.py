class Solution:
    def minimumChairs(self, s: str) -> int:
        chairs_needed = 0
        max_people = 0
        people = 0
        
        for event in s:
            if event == 'E':
                people += 1
                if people > max_people:
                    max_people = people
            elif event == 'L':
                people -= 1
        
        chairs_needed = max_people
        return chairs_needed