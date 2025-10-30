import heapq
import sys

def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())

    # Initialize noodles received for each person (1-indexed list)
    # Size N+1, index 0 is unused
    noodles_received = [0] * (N + 1)

    # Initialize min-heap for people currently in the row (by person ID)
    # Initially, people 1 to N are in the row, sorted by ID.
    # A min-heap naturally keeps the smallest ID at the top, which represents the front of the line
    # according to the interpretation derived from the sample.
    people_present_heap = list(range(1, N + 1))
    heapq.heapify(people_present_heap)

    # Initialize event queue (min-heap)
    # Events are tuples: (time, type_code, ...)
    # type_code 0: return event (time, 0, person_id)
    # type_code 1: noodle event (time, 1, quantity, return_duration)
    # Using type_code ensures return events are prioritized at the same time
    event_heap = []

    # Add initial noodle events to the event queue
    for _ in range(M):
        T, W, S = map(int, sys.stdin.readline().split())
        # Store (time, type=1, quantity, return_duration)
        heapq.heappush(event_heap, (T, 1, W, S))

    # Process events from the event queue chronologically
    while event_heap:
        # Get the next event with the minimum time
        current_time, event_type, *event_data = heapq.heappop(event_heap)

        if event_type == 0: # Return event
            person_id = event_data[0]
            # Add the person back to the queue of present people
            # Adding to the min-heap maintains the property that the smallest ID is at the front
            heapq.heappush(people_present_heap, person_id)

        elif event_type == 1: # Noodle event
            quantity, return_duration = event_data
            
            # Check if there are people currently in the row
            if people_present_heap:
                # The person at the front is the one with the smallest original ID
                # Pop the smallest ID from the heap
                person_at_front = heapq.heappop(people_present_heap)

                # Give noodles to this person
                noodles_received[person_at_front] += quantity

                # Schedule the person's return to the row
                return_time = current_time + return_duration
                # Push the return event: (return_time, type=0, person_id)
                heapq.heappush(event_heap, (return_time, 0, person_at_front))

    # Print the total noodles received by each person (1 to N)
    for i in range(1, N + 1):
        sys.stdout.write(str(noodles_received[i]) + '
')

# Execute the solve function
solve()