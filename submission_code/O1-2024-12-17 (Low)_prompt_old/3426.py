class Solution:
    def minimumChairs(self, s: str) -> int:
        current_people = 0
        max_chairs = 0
        
        for char in s:
            if char == 'E':
                current_people += 1
                max_chairs = max(max_chairs, current_people)
            else:  # char == 'L'
                current_people -= 1
                
        return max_chairs