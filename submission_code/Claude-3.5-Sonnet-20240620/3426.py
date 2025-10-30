class Solution:
    def minimumChairs(self, s: str) -> int:
        chairs_needed = 0
        current_chairs = 0
        
        for event in s:
            if event == 'E':
                if current_chairs == 0:
                    chairs_needed += 1
                    current_chairs += 1
                else:
                    current_chairs -= 1
            else:  # event == 'L'
                current_chairs += 1
        
        return chairs_needed