# YOUR CODE HERE
import heapq
import sys

# Function to solve the problem
def solve():
    # Read N (number of people) and M (number of noodle events) from input
    # Use sys.stdin.readline for faster input reading compared to input()
    N, M = map(int, sys.stdin.readline().split())

    # Initialize a list to store the total amount of noodles received by each person.
    # Use 1-based indexing for people (1 to N), so the list size is N+1. Index 0 is unused.
    # Initialize all totals to 0.
    total_noodles = [0] * (N + 1)

    # Initialize a set to keep track of the indices of people currently in the row.
    # A set provides fast (average O(1)) lookups, additions, and removals.
    available_set = set(range(1, N + 1)) 
    
    # Initialize a min-heap to store the indices of people currently in the row.
    # The min-heap allows efficiently finding the person with the minimum index, 
    # who is considered to be at the front of the row.
    # Start with all people from 1 to N in the heap.
    available_heap = list(range(1, N + 1)) 
    heapq.heapify(available_heap) # Convert the list into a valid min-heap in O(N) time.

    # Initialize a priority queue (min-heap) to manage events.
    # Events are stored as tuples: (time, type_priority, data).
    # - time: The time the event occurs.
    # - type_priority: An integer to handle tie-breaking for events at the same time. 
    #   We use 0 for RETURN events and 1 for NOODLE events. Lower value means higher priority.
    #   This ensures RETURN events are processed before NOODLE events at the same time `T`, 
    #   as required by the rule "A person who returns to the row at time X is considered to be in the row at time X".
    # - data: Information specific to the event type.
    event_pq = [] 

    # Read the M noodle events from input.
    for _ in range(M):
        T, W, S = map(int, sys.stdin.readline().split())
        # Add each noodle event to the priority queue.
        # For a NOODLE event, the data tuple is (W, S), representing the quantity of noodles 
        # and the duration the person stays out of the row after receiving noodles.
        heapq.heappush(event_pq, (T, 1, (W, S))) 

    # Process events in chronological order until the priority queue is empty.
    while event_pq:
        # Extract the event with the smallest time (and highest priority in case of tie).
        time, type_priority, data = heapq.heappop(event_pq)

        if type_priority == 1: # This is a NOODLE event
            W, S = data # Unpack noodle quantity (W) and stay-out duration (S) from event data.
            
            # Find the person currently at the front of the row.
            # This is the person with the minimum index among those currently in `available_set`.
            # We use a "lazy deletion" strategy with `available_heap`: elements are only physically
            # removed from the heap when they are popped. We check if the popped element is valid 
            # (i.e., present in `available_set`) before processing it.
            person_to_serve = -1 # Initialize to -1, indicating no one found yet or row might be empty.
            
            # Repeatedly pop the minimum element from `available_heap` until we find one
            # that is currently present in `available_set` or the heap becomes empty.
            while available_heap:
                # Get the person ID with the smallest value currently at the top of the heap.
                candidate = heapq.heappop(available_heap)
                
                # Check if this candidate person is actually available (present in `available_set`).
                if candidate in available_set:
                    # Yes, this is the person at the front. Record their ID.
                    person_to_serve = candidate
                    break # Exit the loop, we have found the person at the front.
                # else: The popped candidate is not in `available_set`. 
                # This means they left the row earlier and haven't returned yet. 
                # This entry in the heap is stale/invalid. We discard it and continue the loop
                # to check the next smallest element from the heap.
            
            # Check if we successfully found a person at the front.
            if person_to_serve != -1:
                # Add the received noodle quantity W to this person's total count.
                total_noodles[person_to_serve] += W
                
                # The person now steps out of the row. Update their availability status.
                # Remove them from the set of available people.
                available_set.remove(person_to_serve) 
                # Note: `person_to_serve` was already physically removed from `available_heap` 
                # by the `heapq.heappop` call inside the while loop. There's no need to remove it again.

                # Schedule the RETURN event for this person. They will return at time `time + S`.
                return_time = time + S
                # For a RETURN event, the data is simply the ID of the person returning.
                heapq.heappush(event_pq, (return_time, 0, person_to_serve))
            # Else (person_to_serve == -1): 
            # The while loop finished without finding any person from the heap that was also in `available_set`.
            # This can happen if `available_heap` became empty or all remaining elements were stale.
            # This signifies the row is currently empty. No one receives noodles for this event. Do nothing.

        else: # This is a RETURN event (type_priority == 0)
            person_id = data # Get the ID of the person returning from the event data.
            
            # The person returns to the row.
            # A check `if person_id not in available_set:` could be added as a safeguard,
            # but based on the problem logic, a person only returns after they have left,
            # so they should not be in `available_set` at the time of their RETURN event.
            # Mark the person as available again by adding their ID back to the set.
            available_set.add(person_id)
            # Add the person's ID back to the min-heap. They are now eligible to be at the front 
            # of the row again if they have the minimum ID among available people.
            heapq.heappush(available_heap, person_id)

    # After processing all events in the priority queue, the simulation is complete.
    # Print the final results: the total amount of noodles received by each person.
    # Output N lines, where the i-th line contains the total for person i.
    for i in range(1, N + 1):
        print(total_noodles[i])

# Execute the main function to solve the problem.
solve()