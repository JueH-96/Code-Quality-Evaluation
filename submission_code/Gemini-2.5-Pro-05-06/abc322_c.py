import sys

def solve():
    # Read N (total days) and M (number of firework days)
    N, M = map(int, sys.stdin.readline().split())
    
    # Read the list A of firework days.
    # A contains 1-based day numbers, sorted in increasing order.
    # The last firework day A[M-1] is guaranteed to be N.
    A = list(map(int, sys.stdin.readline().split()))

    # firework_event_ptr is an index for the list A.
    # It points to the next relevant firework day we are considering.
    firework_event_ptr = 0
    
    # Iterate for each day from 1 to N
    for current_day in range(1, N + 1):
        # We need to find the first firework day A[firework_event_ptr]
        # such that A[firework_event_ptr] >= current_day.
        # If A[firework_event_ptr] < current_day, it means the firework event
        # at A[firework_event_ptr] has already passed relative to current_day.
        # So, we advance firework_event_ptr to consider the next firework event.
        # This loop is safe from IndexError because A[M-1] = N, and current_day <= N.
        # If firework_event_ptr becomes M-1, A[firework_event_ptr] = N.
        # The condition N < current_day (for current_day <= N) will be false,
        # so firework_event_ptr will not be incremented beyond M-1.
        while A[firework_event_ptr] < current_day:
            firework_event_ptr += 1
            
        # Now, A[firework_event_ptr] is the day number of the first
        # firework event occurring on or after current_day.
        # The number of days to wait is A[firework_event_ptr] - current_day.
        days_to_wait = A[firework_event_ptr] - current_day
        
        # Print the result for current_day.
        # sys.stdout.write requires a string argument and a newline character.
        sys.stdout.write(str(days_to_wait) + "
")

# Call the solver function.
solve()