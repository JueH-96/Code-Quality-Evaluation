# YOUR CODE HERE
import sys
import heapq
from sortedcontainers import SortedList

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # person_noodles[i] will store the total amount of noodles person i has received.
    # We use 1-based indexing for people, so the list size is N+1.
    person_noodles = [0] * (N + 1)

    # available_people: A SortedList to store the IDs of people currently in the row.
    # SortedList automatically keeps elements in sorted order (smallest ID at index 0),
    # and allows efficient O(log K) addition and removal (K is list size).
    # Initially, all N people are in the row, in order 1 to N.
    available_people = SortedList(range(1, N + 1))

    # returning_events: A min-heap to manage people who have left the row and are scheduled to return.
    # Each element in the heap is a tuple (return_time, person_id).
    # The heap ensures that the person returning earliest is always at the top.
    returning_events = []

    # Read all M noodle flow events. They are already guaranteed to be sorted by time T_i.
    noodle_flow_data = []
    for _ in range(M):
        T, W, S = map(int, sys.stdin.readline().split())
        noodle_flow_data.append((T, W, S))

    # Process each noodle flow event chronologically
    for T_i, W_i, S_i in noodle_flow_data:
        # Step 1: Before processing the current noodle event at T_i,
        # add all people whose return_time is <= T_i back to the `available_people` list.
        # This loop efficiently processes all returns that happen up to and including T_i.
        while returning_events and returning_events[0][0] <= T_i:
            return_time, person_id = heapq.heappop(returning_events)
            available_people.add(person_id)

        # Step 2: Process the current noodle flow event at T_i.
        # Check if there's anyone currently in the row (i.e., available_people is not empty).
        if available_people:
            # The person at the front of the row is the one with the smallest person_id.
            # In SortedList, this is available_people[0]. pop(0) removes it efficiently.
            current_front_person_id = available_people.pop(0)
            
            # Give the noodles to this person.
            person_noodles[current_front_person_id] += W_i
            
            # Schedule this person's return: they leave the row and will return at T_i + S_i.
            # Add them to the returning_events min-heap.
            heapq.heappush(returning_events, (T_i + S_i, current_front_person_id))
        # If available_people is empty, no one is currently in the row, so the noodles are lost.

    # After all M events are processed, print the total amount of noodles each person has received.
    for i in range(1, N + 1):
        sys.stdout.write(str(person_noodles[i]) + "
")

# The standard entry point for competitive programming platforms.
if __name__ == '__main__':
    solve()