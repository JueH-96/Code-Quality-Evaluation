class Solution:
    def minimumChairs(self, s: str) -> int:
        """
        Calculates the minimum number of chairs required for a waiting room 
        based on a sequence of entries ('E') and leaves ('L').

        The minimum number of chairs needed is determined by the maximum number
        of people present in the waiting room at any given time during the sequence 
        of events. We simulate the process step by step, keeping track of the 
        current number of people and the maximum number encountered so far.

        Args:
          s: A string where each character represents an event at a second.
             'E' denotes a person entering the room.
             'L' denotes a person leaving the room.
             The input string `s` is guaranteed to represent a valid sequence, 
             meaning the number of people in the room never becomes negative.
             The room starts empty.

        Returns:
          The minimum number of chairs required to accommodate everyone entering,
          which is equal to the peak number of people simultaneously in the room.
        """
        
        current_people = 0 # Initialize count of people currently in the room
        max_people = 0     # Initialize maximum number of people seen simultaneously

        # Simulate the events second by second by iterating through the string
        for event in s:
            if event == 'E':
                # A person enters, increment the current count of people.
                current_people += 1
                # A chair is needed for this entering person. The total number of chairs
                # must be at least the current number of people. We update the maximum
                # number of people seen so far, as this determines the minimum chairs required.
                max_people = max(max_people, current_people)
            elif event == 'L':
                # A person leaves, decrement the current count of people.
                # The problem statement guarantees that `s` represents a valid sequence,
                # which implies that a person only leaves if there is someone in the room
                # (i.e., current_people >= 1 before this decrement).
                current_people -= 1

        # After processing all events, max_people holds the peak occupancy encountered
        # during the simulation. This peak value represents the minimum number of
        # chairs needed to ensure a chair is available for every entering person.
        return max_people