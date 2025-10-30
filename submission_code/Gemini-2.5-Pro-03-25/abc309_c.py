# YOUR CODE HERE
import sys

def solve():
    # Read N (number of medicine kinds) and K (pill limit) from standard input
    N, K = map(int, sys.stdin.readline().split())
    
    # Dictionary to store events: 
    # Key: day `a` when a prescription ends.
    # Value: total `b` pills from all prescriptions ending exactly on day `a`.
    # This dictionary will aggregate the total reduction in pills that occurs on day `a + 1`.
    events = {}
    
    # Variable to store the total number of pills taken on the current day segment being considered.
    # We initialize by calculating the total pills for Day 1.
    current_pills = 0
    
    # Read N prescription details (a_i, b_i) from standard input
    for _ in range(N):
        # a: number of days to take medicine (medicine i is taken on days 1 through a)
        # b: number of pills of medicine i per day
        a, b = map(int, sys.stdin.readline().split())
        
        # Add to the initial total pills count for Day 1.
        # All prescriptions are active on Day 1, so sum all b_i.
        current_pills += b
        
        # Record the reduction event: 
        # On day `a+1`, the number of pills taken decreases by `b` because the prescription for this medicine ends after day `a`.
        # We store this information by mapping the expiration day `a` to the pill amount `b`.
        # The `events.get(a, 0)` part handles the case where multiple prescriptions might end on the same day `a`. 
        # It retrieves the current sum for day `a` (or 0 if none exists yet) and adds `b` to it.
        events[a] = events.get(a, 0) + b
        
    # Check if the condition (total pills <= K) is met on Day 1 itself.
    # If `current_pills` (total pills on Day 1) is already less than or equal to K,
    # then Day 1 is the first day the condition is met.
    if current_pills <= K:
        print(1)
        # Exit the function after printing the result.
        return

    # Get the unique expiration days (`a` values which are keys in the events dictionary) 
    # and sort them in ascending order. This gives us the chronological order of events.
    sorted_days = sorted(events.keys())
    
    # Iterate through the sorted expiration days `d`.
    # `current_pills` holds the total pills count for the days *before* the reduction associated with `d` happens.
    # Specifically, it's the count for days from (previous expiration day + 1) up to `d`.
    for day in sorted_days:
        # Retrieve the total number of pills to be removed from the daily count
        # due to prescriptions ending on this `day`. This reduction takes effect on `day + 1`.
        pills_to_remove = events[day]
        
        # Update the total pills count. `current_pills` now reflects the number of pills taken
        # per day starting from `day + 1` onwards, until the next expiration event occurs.
        current_pills -= pills_to_remove
        
        # Check if the updated pill count satisfies the condition (<= K).
        if current_pills <= K:
            # If the condition is met (`current_pills` <= K), this means that starting from day `day + 1`,
            # the total number of pills taken is K or less. Since we are processing days in chronological order,
            # `day + 1` is the *first* day this condition holds true (after Day 1, which we already checked).
            # Print the result (`day + 1`) to standard output.
            print(day + 1)
            # Terminate the program as we have found the answer.
            return

# Execute the main logic encapsulated in the solve function.
# This function reads input, performs calculations, and prints the output.
solve()