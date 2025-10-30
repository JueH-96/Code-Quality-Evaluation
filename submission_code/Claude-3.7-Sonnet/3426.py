class Solution:
    def minimumChairs(self, s: str) -> int:
        current_occupancy = 0  # Current number of people in the room
        max_occupancy = 0      # Maximum number of people ever in the room
        
        for char in s:
            if char == 'E':
                # A person enters and takes a chair
                current_occupancy += 1
                # Update the maximum if needed
                max_occupancy = max(max_occupancy, current_occupancy)
            else:  # char == 'L'
                # A person leaves, freeing up a chair
                current_occupancy -= 1
        
        return max_occupancy