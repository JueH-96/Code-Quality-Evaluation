import sys

def solve():
    # Read N (number of people) and A (time per purchase)
    N, A = map(int, sys.stdin.readline().split())
    
    # Read T_values (list of arrival times)
    # T_values are already sorted: T_1 < T_2 < ... < T_N
    T_values_str = sys.stdin.readline().split()
    T_values = [int(t) for t in T_values_str]

    # This variable tracks the time when the ticket booth becomes free.
    # Initially, the booth is free at time 0.
    booth_free_time = 0 

    # Iterate through each person's arrival time
    for person_arrival_time in T_values:
        # A person can start buying a ticket either when they arrive,
        # or when the booth becomes free, whichever is later.
        purchase_start_time = max(person_arrival_time, booth_free_time)
        
        # The purchase takes A seconds.
        purchase_finish_time = purchase_start_time + A
        
        # Output the finish time for this person.
        print(purchase_finish_time)
        
        # The booth will now be free at this new finish time, ready for the next person.
        booth_free_time = purchase_finish_time

if __name__ == "__main__":
    solve()