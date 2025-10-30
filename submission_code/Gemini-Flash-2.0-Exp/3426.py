class Solution:
    def minimumChairs(self, s: str) -> int:
        chairs_needed = 0
        current_occupancy = 0
        for event in s:
            if event == 'E':
                current_occupancy += 1
                chairs_needed = max(chairs_needed, current_occupancy)
            else:
                current_occupancy -= 1
        return chairs_needed