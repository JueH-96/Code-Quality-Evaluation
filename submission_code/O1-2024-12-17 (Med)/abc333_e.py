def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast I/O parsing
    N = int(input_data[0])
    tx = input_data[1:]  # length 2*N

    # t[i], x[i] for i in [1..N], 1-based indexing
    # We'll store them in arrays for convenience:
    t = [0]*(N+1)
    x = [0]*(N+1)
    idx = 0
    for i in range(1, N+1):
        t[i] = int(tx[idx]); x[i] = int(tx[idx+1])
        idx += 2

    # We want to determine if Takahashi can survive all monsters.
    # If not, print -1.
    # Otherwise, let K be the maximum number of potions carried at any point
    # (i.e., the maximum inventory size during the adventure).
    # We want to minimize K over all valid strategies. We also must output
    # one valid picking strategy (0/1 for each potion event) that achieves
    # this minimal K.

    # ----------------------------
    # STRATEGY (Greedy "only pick if it is strictly needed"):
    #
    # 1) Count total monsters of each type to know how many are needed overall.
    # 2) Collect all potion positions by type so we can quickly find how many potions
    #    remain "in the future" at each step (by binary search).
    # 3) We'll do one forward pass (i=1..N), maintaining:
    #       - needed[x]: how many monsters of type x are still not yet faced (or not yet matched).
    #         Actually we'll decrement it each time we face a monster, and each time that monster
    #         is matched with a potion in-hand.
    #       - a stack (or list) of indices of potions in current inventory for each type x.
    #       - a global current_inventory = sum of all stacks' sizes.
    #    When we see a potion event (t_i=1):
    #        we check if we must pick it:
    #           condition: (needed[x] > (number of potions of type x that appear after i) + stack_x.size())
    #           If true, we pick it (push onto stack_x, assignment=1, current_inventory++)
    #           else skip it (assignment=0).
    #    When we see a monster event (t_i=2):
    #        we must have at least one potion in stack_x; if empty => fail (-1).
    #        pop one potion from stack_x, current_inventory--, and that covers this monster.
    #        also decrement needed[x] by 1.
    #    We track the maximum of current_inventory at each step for K.
    #
    # Correctness intuition:
    # - We never pick a potion unless it is strictly needed to cover the remaining monsters
    #   beyond what we already hold plus what will appear in the future.
    # - This ensures we do not inflate inventory with unneeded potions.
    # - Using the earliest potions in the stack for monsters does not change the size of
    #   the overall inventory at each step (a pop is a pop), so any LIFO/FIFO choice per type
    #   does not affect the total count; we just need to be consistent and pop one.
    #
    # After the pass, if we never failed, we have a valid pick/skip assignment.
    # The recorded max_inventory is then the minimal possible K. Output that and the picks.

    # Step 1) Count how many monsters of each type; also track potions of each type.
    from collections import defaultdict
    potions_positions = defaultdict(list)  # x -> sorted list of indices i where t_i=1 and x_i=x
    needed = defaultdict(int)              # x -> total number of monsters (we will decrement on usage)

    # Tally total monsters
    for i in range(1, N+1):
        if t[i] == 2:
            needed[x[i]] += 1

    # Build the list of potion positions
    # We'll do it in ascending order of i
    for i in range(1, N+1):
        if t[i] == 1:
            potions_positions[x[i]].append(i)

    # Precompute for each type x how to find "how many potions appear after index i".
    # We will do a binary search on potions_positions[x].
    import bisect
    def count_potions_after(typ, pos):
        # returns how many potion events for 'typ' occur at an index > pos
        arr = potions_positions[typ]
        # we want the pointer to the right of 'pos', so use bisect_right
        # idx is how many are <= pos, so potions strictly greater than pos is len - idx
        idx = bisect.bisect_right(arr, pos)
        return len(arr) - idx

    # Step 2) We'll do one forward pass. For each type x, maintain a stack of potions that we have picked.
    # assignment[i] in {0,1} for i where t_i=1. Default is 0.
    assignment = [0]*(N+1)
    stacks = defaultdict(list)    # type x -> stack of indices of potions we are currently holding
    current_inventory = 0
    max_inventory = 0

    # We'll also keep track how many monsters are left for x: that is needed[x].
    # Each time we face a monster event of type x, we must pop from the stack (if empty => fail).
    # Then needed[x] -= 1
    #
    # Each time we see a potion event of type x at i, we check if needed[x] > (count_potions_after(x, i) + len(stacks[x]))
    # If true => pick the potion => assignment[i]=1, push i to stacks[x], current_inventory++.
    # else => skip => assignment[i]=0.

    for i in range(1, N+1):
        if t[i] == 1:
            typ = x[i]
            # If we still have monsters left for this type, decide if we must pick
            if needed[typ] > count_potions_after(typ, i) + len(stacks[typ]):
                # pick
                assignment[i] = 1
                stacks[typ].append(i)
                current_inventory += 1
            else:
                # skip
                assignment[i] = 0

        else:
            # monster event
            typ = x[i]
            if len(stacks[typ]) == 0:
                # no potion in hand for this type => fail
                print(-1)
                return
            # use the oldest or newest doesn't affect the total size; pop one
            stacks[typ].pop()
            current_inventory -= 1
            needed[typ] -= 1

        # update max inventory
        if current_inventory > max_inventory:
            max_inventory = current_inventory

    # If we got here, we managed to defeat all monsters without failing
    # The minimal possible maximum number of potions carried (K_min) is max_inventory.
    print(max_inventory)

    # Finally, we must output the actions for each event i with t_i=1 in ascending order i:
    #  1 if we picked that potion, 0 if we skipped it.
    picks = []
    for i in range(1, N+1):
        if t[i] == 1:
            picks.append(str(assignment[i]))
    print(" ".join(picks))


# Make sure to call main()
if __name__ == "__main__":
    main()