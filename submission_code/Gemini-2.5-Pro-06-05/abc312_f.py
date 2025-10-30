import sys
import bisect
from itertools import accumulate

def solve():
    """
    Reads input, solves the item selection problem, and prints the result.
    """
    # Use fast I/O
    input = sys.stdin.readline
    
    try:
        # Read problem parameters N and M
        line = input()
        if not line: return
        N, M = map(int, line.split())
    except (IOError, ValueError):
        return

    # Categorize items
    pull_tabs = []
    regular_cans = []
    openers = []
    
    for _ in range(N):
        try:
            line = input()
            if not line: break
            T, X = map(int, line.split())
            if T == 0:
                pull_tabs.append(X)
            elif T == 1:
                regular_cans.append(X)
            else:  # T == 2
                openers.append(X)
        except (IOError, ValueError):
            break
            
    # Sort each category in descending order to prioritize high-value items
    pull_tabs.sort(reverse=True)
    regular_cans.sort(reverse=True)
    openers.sort(reverse=True)
    
    # Calculate prefix sums for efficient lookups
    # A '0' is prepended for convenient 1-based indexing of sums
    P_pull = [0] + list(accumulate(pull_tabs))
    P_reg = [0] + list(accumulate(regular_cans))
    P_open = [0] + list(accumulate(openers))
    
    max_happiness = 0
    
    # Initialize 'k' (number of regular cans) to the max possible value
    k = min(M, len(regular_cans))

    # Iterate on 'i', the number of pull-tab cans to take
    for i in range(min(M, len(pull_tabs)) + 1):
        
        # Slots remaining for regular cans and openers
        slots_left = M - i
        if slots_left < 0:
            break
        
        # Since slots_left decreases, k can't increase.
        # We can cap k to not exceed the remaining slots.
        k = min(k, slots_left)

        # Shrink 'k' until it's a valid choice for the remaining slots
        while k > 0:
            # Find minimum openers 'j' needed for capacity 'k'
            j = bisect.bisect_left(P_open, k)
            
            # Check feasibility:
            # 1. We must have enough openers (j <= len(openers))
            # 2. The total items (k cans + j openers) must fit in slots_left
            if j <= len(openers) and k + j <= slots_left:
                break
            
            k -= 1

        # Calculate happiness for this combination of i pull-tabs and k regular cans
        current_happiness = P_pull[i] + P_reg[k]
        max_happiness = max(max_happiness, current_happiness)
            
    print(max_happiness)

solve()