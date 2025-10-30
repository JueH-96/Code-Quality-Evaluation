class Solution:
    def minimumChairs(self, s: str) -> int:
        current_occupancy = 0
        max_occupancy = 0
        
        for event in s:
            if event == 'E':
                current_occupancy += 1
            elif event == 'L':
                # According to the problem: "s represents a valid sequence of entries and exits."
                # This implies that a person only leaves if there's someone in the room.
                # Therefore, current_occupancy will not go below 0 if it was positive before the 'L' event.
                current_occupancy -= 1
            
            # The minimum number of chairs needed is determined by the peak number
            # of people simultaneously in the waiting room.
            # We update max_occupancy if the current_occupancy has reached a new peak.
            # Note: if current_occupancy decreases due to 'L', it won't exceed max_occupancy
            # unless max_occupancy was based on a previous, lower peak (not possible here as it only grows).
            # max() handles this correctly.
            max_occupancy = max(max_occupancy, current_occupancy)
                
        return max_occupancy