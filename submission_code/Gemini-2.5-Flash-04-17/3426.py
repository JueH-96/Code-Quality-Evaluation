class Solution:
    def minimumChairs(self, s: str) -> int:
        """
        Calculates the minimum number of chairs needed in a waiting room
        based on a sequence of entry and exit events.

        Args:
            s: A string representing the sequence of events ('E' for enter, 'L' for leave).

        Returns:
            The minimum number of chairs required.
        """
        current_occupancy = 0  # Number of people currently in the waiting room
        max_chairs_needed = 0  # Maximum number of people observed at any point

        for event in s:
            if event == 'E':
                # A person enters, increment the current occupancy.
                current_occupancy += 1
                # The minimum number of chairs needed is the maximum occupancy
                # reached at any point. Update the max if current is higher.
                max_chairs_needed = max(max_chairs_needed, current_occupancy)
            elif event == 'L':
                # A person leaves, decrement the current occupancy.
                # The problem constraints imply current_occupancy will not go below zero.
                current_occupancy -= 1

        return max_chairs_needed