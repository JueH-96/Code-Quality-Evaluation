import sys
import heapq

# Set up fast I/O
input = sys.stdin.readline

def solve():
    """
    Solves the Flowing Noodles problem by simulating the events using heaps.
    """
    # Read problem parameters
    try:
        N, M = map(int, input().split())
    except (IOError, ValueError):
        # Handle empty input case
        return

    # Initialize an array to store the total amount of noodles for each person.
    # ans[i] corresponds to person i+1.
    ans = [0] * N

    # `available_people` is a min-heap storing the IDs of people currently in the line.
    # The person with the smallest ID is considered to be at the front of the line.
    # Initially, all N people are in the line in order.
    available_people = list(range(1, N + 1))
    heapq.heapify(available_people)  # O(N) operation to establish heap invariant.

    # `returning_people` is a min-heap storing tuples of (return_time, person_id).
    # This allows us to efficiently find the next person to return to the line.
    returning_people = []

    # Process the M events in the given chronological order.
    for _ in range(M):
        # Read the details of the current event.
        T, W, S = map(int, input().split())

        # Before handling the event at time T, process all people who should have
        # returned to the line by this time.
        # A person returning at time X is considered to be in the row at time X.
        while returning_people and returning_people[0][0] <= T:
            # Get the person with the earliest return time from the heap.
            _return_time, person_id = heapq.heappop(returning_people)
            
            # Add this person back to the line of available people.
            heapq.heappush(available_people, person_id)

        # Now, handle the noodle event at time T.
        # Check if there is anyone in the line.
        if available_people:
            # The person at the front of the line is the one with the smallest ID.
            person_at_front = heapq.heappop(available_people)
            
            # This person receives the noodles.
            ans[person_at_front - 1] += W
            
            # The person steps out of the line and is scheduled to return.
            return_time = T + S
            heapq.heappush(returning_people, (return_time, person_at_front))
        # If `available_people` is empty, the noodles are wasted.

    # After all M events have been processed, print the results.
    for total_noodles in ans:
        print(total_noodles)

# Run the solution
solve()