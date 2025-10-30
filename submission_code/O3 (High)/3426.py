class Solution:
    def minimumChairs(self, s: str) -> int:
        # Current number of people in the waiting room
        current = 0
        # Maximum number of people observed at the same time
        max_people = 0
        
        for ch in s:
            if ch == 'E':         # A person enters
                current += 1
            else:                 # 'L' - a person leaves
                current -= 1
            # Update maximum occupancy
            max_people = max(max_people, current)
            
        return max_people