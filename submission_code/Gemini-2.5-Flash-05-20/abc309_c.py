# YOUR CODE HERE
import sys
import collections

def solve():
    # Read N and K from standard input
    N, K = map(int, sys.stdin.readline().split())
    
    # day_changes: a Counter to store the net change in pills on specific days.
    # The key is the day, and the value is the change in pills (negative for pills removed).
    # A medicine (a_i, b_i) is taken from day 1 to day a_i.
    # This means its contribution of b_i pills stops on day a_i + 1.
    day_changes = collections.Counter()
    
    # Calculate the initial total pills taken on Day 1.
    # All medicines are taken on Day 1 because a_i >= 1.
    initial_total_pills = 0
    
    for _ in range(N):
        a_i, b_i = map(int, sys.stdin.readline().split())
        initial_total_pills += b_i
        # On day a_i + 1, b_i pills are no longer taken.
        day_changes[a_i + 1] -= b_i

    # Check if Day 1 itself satisfies the condition.
    if initial_total_pills <= K:
        print(1)
        return

    # Initialize current_pills with the total for Day 1.
    current_pills = initial_total_pills
    
    # Get all unique days where pill counts change, and sort them.
    # These are the 'event points' for our sweep line.
    sorted_change_days = sorted(day_changes.keys())
    
    # last_processed_day keeps track of the first day in the current segment
    # for which `current_pills` is the valid pill count.
    # Initially, `current_pills` is valid from Day 1.
    last_processed_day = 1 

    # Iterate through the sorted event days.
    for next_event_day in sorted_change_days:
        # Check the segment of days [last_processed_day, next_event_day - 1].
        # In this segment, the pill count was constant at `current_pills`.
        # If `next_event_day` is strictly greater than `last_processed_day`, it means
        # we are about to move to a new segment of days. Before processing the change
        # for `next_event_day`, `current_pills` represents the total for `last_processed_day`
        # and subsequent days up to `next_event_day - 1`.
        # Since we seek the *first* day, if `current_pills` is <= K, then `last_processed_day`
        # is the answer.
        if next_event_day > last_processed_day:
            if current_pills <= K:
                print(last_processed_day)
                return
        
        # Apply the pill change that occurs on `next_event_day`.
        current_pills += day_changes[next_event_day]
        
        # Update `last_processed_day` to `next_event_day`.
        # The `current_pills` value (after applying changes) is now valid
        # for `next_event_day` and all subsequent days until the next event.
        last_processed_day = next_event_day 
        
    # If the loop finishes without returning, it means `current_pills` was always > K
    # for all segments encountered *before* their respective `next_event_day` changes.
    # After the loop, `current_pills` holds the total pill count for `last_processed_day`
    # (which will be the last `a_i + 1` from `sorted_change_days`) and all days thereafter.
    # At this point, all medicines would have ended, so `current_pills` will be 0.
    # Since K is non-negative, `0 <= K` is always true.
    # Therefore, `last_processed_day` is the first day from which the condition is met.
    if current_pills <= K:
        print(last_processed_day)
        return

# Call the solve function to execute the program
solve()