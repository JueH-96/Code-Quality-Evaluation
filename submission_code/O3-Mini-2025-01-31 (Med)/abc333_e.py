def main():
    import sys
    from collections import defaultdict
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    events = []  # list of (t, x) for each event in order (0-indexed)
    
    # We'll record for each event its type and associated x.
    # Also, for supply (t==1) events, we need to decide later whether to pick up (1) or skip (0).
    for i in range(n):
        t = int(next(it))
        x = int(next(it))
        events.append((t, x))
    
    # Prepare dictionaries mapping each potion type to the indices where
    # a potion is found (t==1) and where a monster appears (t==2).
    type_to_supply = defaultdict(list)
    type_to_monster = defaultdict(list)
    
    for i, (t, x) in enumerate(events):
        if t == 1:
            type_to_supply[x].append(i)
        else:
            type_to_monster[x].append(i)
    
    # Our plan is: for each type x that appears in a monster attack, we must cover
    # each monster event with a potion taken from a supply event that comes before it.
    # To “delay” the pickup as much as possible (so that the potion stays in inventory
    # for as short a time as possible), we match each monster with the latest
    # possible potion-supply event that is available and occurs before the monster.
    #
    # We iterate type‐by‐type. For type x, let S be the (sorted) list of indices of supply events,
    # and M be the list of monster events for that type.
    # If there are more monsters than supply events, the adventure is doomed.
    # Otherwise, we assign each monster (processed in descending order)
    # a supply event (the greatest event index less than the monster’s index that is still free).
    chosen = {}  # global dictionary: key = global event index for t==1, value True means we pick up that potion.
    for typ, monsters in type_to_monster.items():
        supplies = type_to_supply.get(typ, [])
        if len(monsters) > len(supplies):
            sys.stdout.write("-1")
            return
        # Process monsters in descending order, trying to assign the latest possible supply for each.
        si = len(supplies) - 1  # pointer into supplies (which occur in increasing order)
        for m in sorted(monsters, reverse=True):
            # We need a supply event with index strictly less than m.
            while si >= 0 and supplies[si] >= m:
                si -= 1
            if si < 0:
                # If none available, adventure fails.
                sys.stdout.write("-1")
                return
            # Use supplies[si] for this monster.
            chosen[supplies[si]] = True
            si -= 1
    
    # In our strategy we only pick up the potion at supply events that are used to defeat a monster.
    # Now, let’s simulate the adventure to compute the peak inventory
    # (i.e. the maximum number of potions held at any point). We must also check that every monster
    # finds a matching potion available in inventory.
    inv = defaultdict(int)  # inventory count per type
    global_inventory = 0     # total number of potions held
    peak = 0
    for i, (t, x) in enumerate(events):
        if t == 1:
            # Pickup potion only if our matching has chosen to take it.
            if chosen.get(i, False):
                inv[x] += 1
                global_inventory += 1
                if global_inventory > peak:
                    peak = global_inventory
        else:
            # When encountering a monster of type x, we must use one potion of that type.
            if inv[x] > 0:
                inv[x] -= 1
                global_inventory -= 1
            else:
                # Should not happen because our matching guarantees a preceding chosen supply.
                sys.stdout.write("-1")
                return

    # Output the answers.
    # The first line: the minimal possible maximum inventory (K_min).
    # The second line: for each supply event in the order they appear in the input,
    # output "1" if we pick up the potion, "0" otherwise.
    result = []
    for i, (t, x) in enumerate(events):
        if t == 1:
            result.append("1" if chosen.get(i, False) else "0")
    
    sys.stdout.write(str(peak) + "
" + " ".join(result))
    
    
if __name__ == '__main__':
    main()