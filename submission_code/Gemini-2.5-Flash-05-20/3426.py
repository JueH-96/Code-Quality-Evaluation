class Solution:
    def minimumChairs(self, s: str) -> int:
        """
        Calculates the minimum number of chairs needed in a waiting room
        given a sequence of entry ('E') and leave ('L') events.

        The problem specifies that the waiting room is initially empty
        and a chair must be available for every person who enters.
        This means we need to track the maximum number of people
        simultaneously present in the waiting room at any point in time.

        Args:
            s: A string representing the sequence of events. 'E' for enter, 'L' for leave.

        Returns:
            The minimum number of chairs required.
        """
        
        # current_people_in_room tracks the number of people currently occupying chairs.
        # It starts at 0 because the waiting room is initially empty.
        current_people_in_room = 0
        
        # max_people_observed tracks the highest number of people observed in the room
        # at any single point. This value will represent the minimum number of chairs needed.
        max_people_observed = 0
        
        # Iterate through each event in the input string s.
        for event in s:
            if event == 'E':
                # If a person enters, the count of people in the room increases by 1.
                current_people_in_room += 1
            elif event == 'L':
                # If a person leaves, the count of people in the room decreases by 1.
                # The problem constraints ("s represents a valid sequence of entries and exits")
                # ensure that current_people_in_room will not drop below zero in a meaningful way
                # for the maximum calculation, as a person cannot leave if no one is there.
                current_people_in_room -= 1
            
            # After processing each event, update max_people_observed.
            # We take the maximum between the current highest observed count
            # and the number of people currently in the room.
            # This ensures that max_people_observed always holds the peak occupancy.
            max_people_observed = max(max_people_observed, current_people_in_room)
            
        # Once all events have been processed, max_people_observed will contain
        # the highest number of people that were simultaneously in the room.
        # This is exactly the minimum number of chairs required to accommodate everyone.
        return max_people_observed