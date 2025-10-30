class Solution:
    def minimumChairs(self, s: str) -> int:
        current_people = 0
        max_people = 0

        for event in s:
            if event == 'E':
                current_people += 1
                max_people = max(max_people, current_people)
            elif event == 'L':
                current_people -= 1

        return max_people