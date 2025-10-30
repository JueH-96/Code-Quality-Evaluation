import sys

def solve():
    N = int(sys.stdin.readline())
    events = []
    # Store events to iterate in reverse later
    type1_count = 0
    for i in range(N):
        t, x = map(int, sys.stdin.readline().split())
        events.append((t, x))
        if t == 1:
            type1_count += 1

    # Reverse pass to determine necessary pickups and minimum required potions
    # needed_count[x] = number of type x potions required by future monsters
    # that must be sourced from current/past finds (in reverse timeline).
    # This is the minimum number of type x potions required *before* the current event (forward timeline).
    needed_count = [0] * (N + 1) 
    
    # current_needed_sum = sum of needed_count across all types.
    # Represents the total minimum number of potions required *before* the current event (forward timeline).
    current_needed_sum = 0 
    
    # max_needed_sum = maximum value of current_needed_sum encountered during the reverse pass.
    # This will be K_min.
    max_needed_sum = 0 
    
    # actions_reverse_temp will store actions (0 or 1) for type 1 events
    # in the reverse order of their appearance in the input timeline.
    actions_reverse_temp = []
    
    # We process events from N-1 down to 0.
    # We process type 1 events from the last one to the first one.
    # The number of type 1 events is type1_count. The actions_reverse_temp will have this size.

    for i in range(N - 1, -1, -1):
        t, x = events[i]

        if t == 2: # Monster encounter
            # Need one potion of type x for this monster. This increases
            # the minimum requirement propagating backwards.
            needed_count[x] += 1
            current_needed_sum += 1
        elif t == 1: # Potion find
            # We found a potion of type x.
            # If there is a need for a type x potion (needed_count[x] > 0)
            # that must be satisfied by a find at or before this event (in reverse),
            # this potion can satisfy one such need. We must pick it up.
            if needed_count[x] > 0:
                needed_count[x] -= 1
                current_needed_sum -= 1 # This potion reduces a future requirement
                actions_reverse_temp.append(1) # Action for this type 1 event is pick up (1)
            else:
                # This potion is not strictly needed for any future monster
                # that hasn't been covered by an even later potion find.
                # To minimize K, we choose to discard it.
                actions_reverse_temp.append(0) # Action for this type 1 event is discard (0)
            
        # After processing event i (in reverse), current_needed_sum is the
        # total minimum number of potions required just BEFORE event i
        # (in forward time). We track the maximum of this value across all
        # points in time (just before each event).
        max_needed_sum = max(max_needed_sum, current_needed_sum)

    # Check if survival is possible. If any needed_count[x] is > 0 after the
    # reverse pass, it means some monsters required type x potions that were
    # never found before the first event.
    for x in range(1, N + 1):
        if needed_count[x] > 0:
            print(-1)
            return

    # Survival is possible. Print K_min.
    print(max_needed_sum)

    # The actions collected in actions_reverse_temp are in reverse order
    # of the original type 1 events. Reverse the list to get the actions
    # in the original order of type 1 events.
    actions = actions_reverse_temp[::-1]
    print(*actions)

solve()