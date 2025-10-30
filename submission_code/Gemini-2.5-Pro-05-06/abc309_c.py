import sys
from collections import defaultdict

def main():
    N, K = map(int, sys.stdin.readline().split())
    
    # day_to_change_in_pills stores the total change in pill count on a given day.
    # A medicine (a_i, b_i) is taken for a_i days (day 1 to day a_i).
    # So, on day a_i + 1, b_i pills from this medicine are no longer taken.
    # The total pill count effectively decreases by b_i on day a_i + 1.
    day_to_change_in_pills = defaultdict(int)
    
    # total_pills_on_day_1 is the sum of b_i for all medicines.
    total_pills_on_day_1 = 0

    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        total_pills_on_day_1 += b
        day_to_change_in_pills[a + 1] -= b # Note: change is negative (decrease)
    
    # Check day 1
    if total_pills_on_day_1 <= K:
        print(1)
        return

    # Get sorted unique days where pill counts change.
    # These are a_i + 1 values, so all are >= 2 (since a_i >= 1).
    # sorted() on a dictionary sorts its keys.
    event_trigger_days = sorted(day_to_change_in_pills.keys())
    
    # current_total_pills tracks the number of pills for the current period.
    # It starts with the pill count of day 1.
    current_total_pills = total_pills_on_day_1
    
    # Iterate through sorted days where total pill count changes.
    # For the first iteration, current_total_pills holds P(1), which is valid for days
    # in [1, first_trigger_day - 1].
    for day in event_trigger_days:
        # Update pill count to reflect changes effective on this 'day'.
        # Now, current_total_pills is the number of pills taken on 'day'
        # and subsequent days until the next trigger day (if any).
        current_total_pills += day_to_change_in_pills[day]
        
        if current_total_pills <= K:
            # This 'day' is the first day the condition is met because:
            # 1. Pill count on day 1 was > K.
            # 2. Pill count on days [prev_trigger_day, day - 1] was also > K.
            #    (If prev_trigger_day was D_prev, then P(D_prev) was > K. This pill count
            #    held for all days up to day-1. Otherwise, D_prev would have been printed
            #    and the program would have exited.)
            # 3. 'day' is the first day the pill count becomes current_total_pills (which is <= K).
            print(day)
            return

if __name__ == '__main__':
    main()