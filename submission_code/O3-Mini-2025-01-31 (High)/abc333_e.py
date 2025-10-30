def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    events = []
    # Read events: each event is (t, x).
    # t = 1 means potion find (with a potion type x),
    # t = 2 means monster encounter (of type x).
    for _ in range(n):
        t = int(next(it))
        x = int(next(it))
        events.append((t, x))
    
    # decision[i] will hold the decision for the i-th event if t==1:
    #   1 means "pick up" and 0 means "discard".
    # For monster events we will not use it.
    decision = [None] * n
    
    # For each potion type x (range 1..n because 1 <= x <= n), we count how many potions
    # are needed later to defeat upcoming monsters. We use a list 'need' of length n+1.
    need = [0] * (n + 1)
    
    # Process events in reverse (backward). Our idea is that every monster event (t==2)
    # increases the demand for a potion of that type. Then, when we encounter a find event,
    # if there is any outstanding need for that type, we decide to pick it up (assign 1)
    # and reduce the need.
    for i in range(n - 1, -1, -1):
        t, x = events[i]
        if t == 2:
            # Encounter a monster of type x: we will need one potion of that type from a previous find event.
            need[x] += 1
        else:
            # Potion find event of type x.
            if need[x] > 0:
                decision[i] = 1  # Pick up this potion to meet a future monster.
                need[x] -= 1
            else:
                decision[i] = 0  # No demand: skip picking up.
    
    # If there remains any outstanding demand for a potion type, then no valid strategy exists.
    for x in range(1, n + 1):
        if need[x] != 0:
            sys.stdout.write("-1")
            return

    # Now simulate the adventure in forward order with the decisions made.
    # Inventory tracks the number of potions available (by type) that have been picked up and are not yet used.
    inv = [0] * (n + 1)
    current_inventory = 0
    max_inventory = 0
    # For the output, we only need the decisions corresponding to "find" events in their original order.
    actions = []
    
    for i in range(n):
        t, x = events[i]
        if t == 1:
            # Only pick up the potion if our backward matching decided so.
            if decision[i] == 1:
                inv[x] += 1
                current_inventory += 1
                if current_inventory > max_inventory:
                    max_inventory = current_inventory
            actions.append(str(decision[i]))
        else:
            # Monster event: defeat the monster using one potion of the matching type.
            if inv[x] > 0:
                inv[x] -= 1
                current_inventory -= 1
            else:
                # This should not happen since our matching ensured a potion is picked up.
                sys.stdout.write("-1")
                return

    # Print K_min (the minimal maximum inventory during the adventure) on the first line.
    # Then, print the list of decisions for each find event (in order, as a sequence of 1's and 0's).
    sys.stdout.write(str(max_inventory) + "
" + " ".join(actions))

if __name__ == '__main__':
    main()