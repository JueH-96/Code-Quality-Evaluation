import heapq
import sys

# It's good practice for competitive programming to wrap the main logic in a function.
def solve():
    # Read N (number of people) and M (number of events)
    N, M = map(int, sys.stdin.readline().split())

    # noodles_eaten[p] will store the total amount of noodles person p has eaten.
    # We use 1-based indexing for people (person 1 to person N), so array size is N+1.
    # noodles_eaten[0] will be unused. Initialize all to 0.
    noodles_eaten = [0] * (N + 1)

    # available_people_heap is a min-heap storing the IDs of people currently in the row.
    # Person IDs range from 1 to N.
    # Initially, all N people are in the row, in order 1, 2, ..., N.
    # Person 1 is at the front.
    available_people_heap = list(range(1, N + 1)) 
    heapq.heapify(available_people_heap) # Converts the list into a min-heap in O(N) time.
                                         # The smallest ID will be at the top.

    # waiting_to_return_heap is a min-heap storing tuples (return_time, person_id).
    # These are people who have eaten noodles and are temporarily out of the row.
    # They will return at their 'return_time'.
    # The heap is ordered by return_time (earliest first).
    waiting_to_return_heap = [] 

    # Process each of the M events
    for _ in range(M):
        # Read event details: T (time), W (quantity of noodles), S (duration person steps out)
        T, W, S = map(int, sys.stdin.readline().split())

        # Step 1: Process returns.
        # Check if any person in waiting_to_return_heap is scheduled to return by time T.
        # The problem states: "A person who returns to the row at time X is considered to be in the row at time X."
        # So, if return_time <= T, they are back.
        while waiting_to_return_heap and waiting_to_return_heap[0][0] <= T:
            # Get the person with the earliest return_time from the heap.
            # heappop returns the tuple (return_time, person_id).
            # We only need person_id_returning to add them back to available_people_heap.
            # Using _ prefix for unused variable as per convention.
            _return_time_val, person_id_returning = heapq.heappop(waiting_to_return_heap) 
            
            # Add this person back to the set of available people.
            heapq.heappush(available_people_heap, person_id_returning)
        
        # Step 2: Select person to eat noodles.
        # If there are any people in the row (available_people_heap is not empty):
        if available_people_heap:
            # The person at the front of the row gets the noodles.
            # This is the person with the smallest ID among those available.
            # heappop from available_people_heap gives this person.
            person_to_eat = heapq.heappop(available_people_heap)
            
            # Add W noodles to this person's total.
            noodles_eaten[person_to_eat] += W
            
            # This person steps out of the row.
            # They will return at time T + S.
            # Add them to the waiting_to_return_heap with their return details.
            heapq.heappush(waiting_to_return_heap, (T + S, person_to_eat))
        # Else (available_people_heap is empty):
        #   No one is in the row. No one gets noodles. Nothing changes for any person.
    
    # Step 3: After all M events, print the results.
    # For each person i from 1 to N, print the total noodles they have eaten.
    for i in range(1, N + 1):
        sys.stdout.write(str(noodles_eaten[i]) + "
")

# Call the solve function to run the program.
solve()