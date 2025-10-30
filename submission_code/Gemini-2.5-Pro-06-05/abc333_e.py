import sys
from collections import defaultdict

def solve():
    """
    This function solves the Takahashi's Adventure problem by determining
    the minimum possible peak potion count (K_min) and the corresponding
    actions.
    """
    try:
        # Fast I/O for performance
        readline = sys.stdin.readline
        
        N = int(readline())
        if N == 0:
            print(0)
            print()
            return
            
        events = [list(map(int, readline().split())) for _ in range(N)]
    except (IOError, ValueError):
        # Handle cases with malformed or empty input
        print(-1)
        return

    # --- Step 1: Determine the necessary potions to pick up via a reverse pass ---
    
    # needed[x] counts required potions of type x for future monsters.
    needed = defaultdict(int)
    
    # actions[i] stores the decision for the potion event at index i.
    actions = {}

    for i in range(N - 1, -1, -1):
        event_type, potion_type = events[i]
        
        if event_type == 1:  # Found a potion
            if needed[potion_type] > 0:
                # A future monster needs this. We must pick it up.
                actions[i] = 1
                needed[potion_type] -= 1
            else:
                # Not needed for any future monster. Discard to minimize inventory.
                actions[i] = 0
        else:  # Encountered a monster
            # This creates a "need" or "debt" for a potion of this type.
            needed[potion_type] += 1

    # --- Step 2: Check for impossibility ---
    # If any needs remain after checking all events, it's impossible.
    if any(count > 0 for count in needed.values()):
        print(-1)
        return

    # --- Step 3: Calculate K_min and the final action sequence ---
    # Simulate the adventure forward with the optimal actions.
    
    current_potions = 0
    max_potions = 0
    final_actions_list = []

    for i in range(N):
        event_type, potion_type = events[i]
        
        if event_type == 1:
            action = actions[i]
            final_actions_list.append(action)
            if action == 1:
                # Pick up potion: count increases.
                # Update max count immediately after pickup.
                current_potions += 1
                max_potions = max(max_potions, current_potions)
        else: # event_type == 2
            # Use potion: count decreases.
            # The peak count before using the potion has already been recorded.
            current_potions -= 1

    # --- Step 4: Print the results ---
    print(max_potions)
    print(*final_actions_list)

# It's good practice to wrap the main logic in a function
# and call it from the script's entry point.
if __name__ == "__main__":
    solve()