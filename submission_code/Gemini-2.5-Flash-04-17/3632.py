class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        # Initialize max_time, max_index, and prev_time using the first event.
        # The time taken for the first event is its absolute time, as per problem definition.
        max_index = events[0][0]
        max_time = events[0][1]
        prev_time = events[0][1]

        # Iterate from the second event onwards (index 1)
        # The loop range handles the edge case len(events) == 1 correctly (loop is empty).
        for i in range(1, len(events)):
            current_index = events[i][0]
            current_time = events[i][1]

            # Calculate the duration of the current button press
            current_duration = current_time - prev_time

            # Compare the current duration with the maximum duration found so far
            if current_duration > max_time:
                # Found a new longest duration
                max_time = current_duration
                max_index = current_index
            # If the current duration is equal to the maximum duration
            elif current_duration == max_time:
                # Apply the tie-breaking rule: choose the button with the smallest index
                if current_index < max_index:
                    max_index = current_index

            # Update prev_time to the time of the current event for the next iteration
            prev_time = current_time

        # After checking all events, max_index holds the index of the button
        # with the longest single press duration, applying the tie-breaking rule.
        return max_index