import sys
from collections import defaultdict

def solve():
    """
    This function reads the input, solves the problem, and prints the result.
    """
    # Read N and K from the first line of standard input.
    # Using sys.stdin.readline for faster I/O with large inputs.
    try:
        n, k = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # This handles cases of empty input, though problem constraints
        # usually guarantee valid input.
        return

    # 'events' will store the total reduction in pills on a given day.
    # The key is the day (a_i + 1), and the value is the sum of b_i for all
    # medicines that stop being taken on that day.
    events = defaultdict(int)
    
    # 'total_pills' will store the sum of pills for day 1.
    total_pills = 0
    
    # Read the N medicine prescriptions.
    for _ in range(n):
        a, b = map(int, sys.stdin.readline().split())
        
        # Add to the total pill count for day 1.
        total_pills += b
        
        # A medicine taken for 'a' days means it is last taken on day 'a'.
        # The pill count decreases on day 'a + 1'.
        events[a + 1] += b
            
    # Check if the pill count on day 1 is already K or less.
    if total_pills <= k:
        print(1)
        return
        
    # Get the unique days on which the pill count changes, and sort them.
    sorted_days = sorted(events.keys())
    
    # 'current_pills' starts as the count on day 1.
    current_pills = total_pills
    
    # Iterate through the event days in chronological order.
    # The pill count is constant between these event days.
    for day in sorted_days:
        # Subtract the pills from medicines that have just expired.
        pills_removed = events[day]
        current_pills -= pills_removed
        
        # If the new pill count is K or less, this is the first day it happens.
        # Since we are processing days in increasing order, this `day` is the answer.
        if current_pills <= k:
            print(day)
            return

# Run the solution.
solve()