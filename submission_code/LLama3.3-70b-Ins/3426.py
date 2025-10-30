class Solution:
    def minimumChairs(self, s: str) -> int:
        chairs = 0
        max_chairs = 0
        people = 0
        
        for event in s:
            if event == 'E':
                people += 1
                chairs += 1
            else:
                people -= 1
                chairs -= 1
            
            # If there are not enough chairs, add more
            if people > chairs:
                chairs = people
            
            # Update the maximum number of chairs needed
            max_chairs = max(max_chairs, chairs)
        
        return max_chairs