import sys

def solve():
    N = int(sys.stdin.readline())
    
    raw_events = []
    for _ in range(N):
        raw_events.append(list(map(int, sys.stdin.readline().split())))
    
    # processed_events stores tuples: (type, potion_x, action_list_idx)
    # action_list_idx is the index in the final 'actions' list if event is type 1.
    # Otherwise, action_list_idx is -1.
    
    processed_events = [] 
    action_list_idx_counter = 0 # Counter for assigning indices to type 1 events

    for i in range(N):
        t, x = raw_events[i]
        if t == 1: # Potion event
            processed_events.append((t, x, action_list_idx_counter))
            action_list_idx_counter += 1
        else: # Monster event
            processed_events.append((t, x, -1)) 
            
    num_potion_events = action_list_idx_counter
    # actions[k] will be 1 if kth potion (in chronological order of finding) is picked up, 0 otherwise.
    actions = [0] * num_potion_events
    
    # required_counts[x] = number of potions of type x that are currently
    # needed by monsters encountered so far in the reverse pass (i.e., monsters at events >= i).
    required_counts = [0] * (N + 1) # Potion types are 1-indexed up to N
    
    current_potions_carried = 0
    max_potions_simultaneously_held = 0

    # Process events in reverse chronological order
    for i in range(N - 1, -1, -1):
        t, x, action_idx = processed_events[i]
        
        if t == 1: # Event i is finding a potion of type x
            if required_counts[x] > 0:
                # This potion is needed for a future monster (from original chronological perspective).
                # Pick it up. This is the latest possible moment to acquire it for that monster.
                actions[action_idx] = 1
                required_counts[x] -= 1
                # This potion is now sourced at event i.
                # It was previously accounted for in current_potions_carried (as needing to be
                # sourced from an event j < i). Since it's now sourced at i, it's no longer
                # counted as one that needs to be carried from an event j < i.
                current_potions_carried -= 1 
            else:
                # This potion is not needed for any monster at k >= i (that isn't already supplied).
                # Discard it to minimize potions held.
                actions[action_idx] = 0 # Default is 0, but explicit for clarity
        
        else: # Event i is encountering a monster of type x
            # This monster needs a potion of type x. This potion must be acquired
            # at some event j < i (chronologically).
            required_counts[x] += 1
            # This required potion adds to the count of potions that must be carried from <i.
            current_potions_carried += 1
            
            # current_potions_carried at this point is the number of potions Takahashi must have
            # in inventory just before event i occurs (chronologically).
            # This is a candidate for K_min.
            if current_potions_carried > max_potions_simultaneously_held:
                max_potions_simultaneously_held = current_potions_carried

    # After iterating through all events in reverse:
    # Check if all potion requirements could be met.
    possible = True
    for x_type_idx in range(1, N + 1): # Potion types are 1 to N
        if required_counts[x_type_idx] > 0:
            # A monster needed potion x_type_idx, but no such potion was found
            # at or before it (chronologically), or all were used for later monsters.
            possible = False
            break
            
    if not possible:
        print("-1")
    else:
        print(max_potions_simultaneously_held)
        # Print actions as space-separated integers
        print(*(str(act) for act in actions))

solve()