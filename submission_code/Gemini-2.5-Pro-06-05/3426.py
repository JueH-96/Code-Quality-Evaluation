class Solution:
    def minimumChairs(self, s: str) -> int:
        """
        Calculates the minimum number of chairs needed in a waiting room based on a sequence of entries and exits.

        Args:
            s: A string representing the sequence of events. 'E' for entry, 'L' for leaving.

        Returns:
            The minimum number of chairs required.
        """
        
        # max_people will store the peak number of people in the room at any time.
        max_people = 0
        
        # current_people tracks the number of people in the room at the current moment.
        current_people = 0
        
        # Iterate through each event in the string s.
        for event in s:
            if event == 'E':
                # A person enters, so the count of people in the room increases.
                current_people += 1
                # The number of chairs needed is determined by the peak occupancy.
                # Update max_people if the current number of people is a new maximum.
                max_people = max(max_people, current_people)
            else:  # event must be 'L'
                # A person leaves, so the count of people in the room decreases.
                current_people -= 1
                
        return max_people