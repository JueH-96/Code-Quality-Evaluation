def main():
    import sys, bisect
    input = sys.stdin.readline

    N = int(input().strip())
    events = []   # list of (t, x)
    # We'll also store indices: 0-indexed events.
    for _ in range(N):
        t, x = map(int, input().split())
        events.append((t, x))
    
    # For potion events, we want to record their positions.
    # For monster events also.
    # We will use 1-indexed types.
    from collections import defaultdict
    potions = defaultdict(list)   # type -> list of indices (0-indexed) for events where t=1
    monsters = defaultdict(list)  # type -> list of indices (0-indexed) for events where t=2
    
    # Also record mapping from overall event index to potion-event index (only if t==1)
    potion_event_map = []   # For each potion event occurrence in order of events, store the global event index.
    # We also want to record the decision later: for potion events, 1 means pick, 0 means skip.
    decision = []
    
    # We'll also need to know the order of potion events (the order in which they appear in the input) because output must be in that order.
    potion_event_global_to_local = dict()  # global_event_index -> local index in decision list.
    
    potion_count = 0
    for i,(t,x) in enumerate(events):
        if t == 1:
            potions[x].append(i)
            potion_event_map.append(i)
            potion_event_global_to_local[i] = potion_count
            decision.append(0)  # default, not picked
            potion_count += 1
        else:
            monsters[x].append(i)
    
    # For each type, we try to assign a potion event (from potions[type]) to each monster event in monsters[type].
    # We store the pairing information in a list of intervals: (pick_event_index, monster_event_index)
    intervals = []
    # For each type:
    for t in monsters:
        # If there is no potion event for that type, it is impossible.
        if len(potions[t]) < len(monsters[t]):
            print(-1)
            return
        # Sort the potion and monster indices.
        # They are already in increasing order by virtue of reading input.
        potion_positions = potions[t]
        monster_positions = monsters[t]
        # We'll do matching backwards; use two pointers:
        pos_ptr = len(potion_positions) - 1
        # Process monsters in reverse order.
        for m_index in reversed(monster_positions):
            # We need a potion event with index < m_index.
            # Because potion_positions is sorted increasing, move pointer leftwards until potion_positions[pos_ptr] < m_index.
            while pos_ptr >= 0 and potion_positions[pos_ptr] >= m_index:
                pos_ptr -= 1
            if pos_ptr < 0:
                # No available potion event to cover this monster. Impossible.
                print(-1)
                return
            assigned_potion = potion_positions[pos_ptr]
            intervals.append((assigned_potion, m_index))
            # Mark that this potion event is used (picked up)
            local_index = potion_event_global_to_local[assigned_potion]
            decision[local_index] = 1
            pos_ptr -= 1  # move pointer to not reuse the same potion event.
    
    # Now we have chosen a set of potion events that will be picked up.
    # Their intervals (from pickup event time to the corresponding monster event time) are stored in intervals.
    # The maximum number of overlapping intervals = the maximum number of potions in inventory at any time.
    # We compute that with a sweep line.
    timeline = []
    # Use events with key value: (time, delta) where delta=+1 for pickup, -1 for used (monster consumption) but to ensure that at the same time a monster event happens,
    # we subtract before adding new potion event? Actually, note: a potion event and a monster event never occur at the same time because they are all in sequential order.
    for (start, end) in intervals:
        # The potion is added at event start, and removed at event end.
        timeline.append((start, 1))
        timeline.append((end, -1))
    timeline.sort()
    current = 0
    peak = 0
    for _, delta in timeline:
        current += delta
        if current > peak:
            peak = current

    # peak is our minimal K
    # Print K and the decisions for each potion event, in order of appearance as they were in input.
    out = []
    out.append(str(peak))
    out.append(" ".join(map(str, decision)))
    sys.stdout.write("
".join(out))
    
if __name__ == '__main__':
    main()