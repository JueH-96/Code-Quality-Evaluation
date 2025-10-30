class Solution:
    def minimumChairs(self, s: str) -> int:
        current_people = 0
        max_people = 0
        for c in s:
            if c == 'E':
                current_people +=1
                if current_people > max_people:
                    max_people = current_people
            elif c == 'L':
                current_people -=1
        return max_people