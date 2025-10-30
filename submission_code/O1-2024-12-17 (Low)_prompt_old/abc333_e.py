def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    tx = input_data[1:]  # pairs t_i, x_i
    
    # Parse the input events
    events = []
    potions_by_type = [[] for _ in range(N+1)]  # potions_by_type[x] = list of indices of potion events with type x
    # We'll also keep track, for each "t=1" event, of its position in an output array
    pick_index_of_event = [-1]*(N+1)  # pick_index_of_event[i] = index in the pick_decisions array if event i is t=1, else -1
    monster_exists = [False]*(N+1)  # to see if there's any monster of type x at all (for quick -1 checks... not strictly necessary)
    
    t_list = [0]*(N+1)
    x_list = [0]*(N+1)
    
    # We'll read the events
    idx_t1 = 0  # counts how many t=1 events have been seen so far, used to assign pick decision indices
    pos = 0
    for i in range(1, N+1):
        t_i = int(tx[pos]); x_i = int(tx[pos+1])
        pos += 2
        t_list[i] = t_i
        x_list[i] = x_i
        if t_i == 1:
            # potion event
            potions_by_type[x_i].append(i)
            pick_index_of_event[i] = idx_t1
            idx_t1 += 1
        else:
            # monster event
            monster_exists[x_i] = True
    
    # If there's a monster of type x but no potions of type x in the entire list, it's automatically impossible:
    # Actually not necessarily if we can't pick any, but we then can't defeat the monster => -1 straight away.
    # But let's wait and see if the logic below fails anyway. This early check is optional.
    
    # We'll do a function that, given K, attempts a simulation to see if we can succeed
    # and if yes, returns the pick_decisions array (of length = total count of t=1 events),
    # otherwise returns None if we fail.
    
    def can_defeat_with_K(K):
        pick_decisions = [0]*idx_t1  # 0 or 1 for each potion event in order of appearance
        # p_x will track how many of the potions of type x we have decided to pick so far
        p_x = [0]*(N+1)
        # used_x will track how many monsters of type x we have defeated (so also how many potions used) so far
        used_x = [0]*(N+1)
        
        bag_size = 0
        max_bag_size = 0
        
        for i in range(1, N+1):
            t_i = t_list[i]
            x_i = x_list[i]
            if t_i == 1:
                # Just skip for now, we only decide to "officially pick" a potion at the moment
                # we need it (just-in-time picking).
                # We'll store the index for possible picking later if needed.
                pass
            else:
                # Monster event
                # We need used_x[x_i] + 1 potions of type x_i to have been picked so far.
                needed = used_x[x_i] + 1
                if p_x[x_i] < needed:
                    # We must pick (needed - p_x[x_i]) more potions of type x_i (in the order they appear)
                    to_pick = needed - p_x[x_i]
                    # Check if we have enough potion events left for type x_i
                    if len(potions_by_type[x_i]) < needed:
                        return None  # not enough potions in total to defeat
                    # We'll pick from the next potions in potions_by_type[x_i], specifically
                    # potions_by_type[x_i][p_x[x_i]], potions_by_type[x_i][p_x[x_i]+1], ...
                    # up to p_x[x_i]+to_pick-1
                    for pick_i in range(to_pick):
                        # pick the potion
                        ev_id = potions_by_type[x_i][ p_x[x_i] + pick_i ]
                        if bag_size + 1 > K:
                            return None
                        bag_size += 1
                        pick_decisions[pick_index_of_event[ev_id]] = 1  # mark that we picked it
                    p_x[x_i] += to_pick
                # now we have at least needed potions of type x_i
                # use one
                used_x[x_i] += 1
                bag_size -= 1  # we remove exactly one potion of type x_i from the bag
                if bag_size < 0:
                    return None  # we can't use a potion if there's none in the bag (shouldn't happen if logic is correct)
            
            # track max bag size if needed
            if bag_size > max_bag_size:
                max_bag_size = bag_size
                if max_bag_size > K:
                    return None
        
        # if we finish successfully, we return the pick_decisions
        return pick_decisions
    
    # We do a quick check: if there's any monster type x that doesn't have enough potions in total to match the number
    # of monsters of that type, it's automatically -1.
    # We'll count how many monster events per type, how many potion events per type
    count_monster = [0]*(N+1)
    count_potion = [0]*(N+1)
    for i in range(1, N+1):
        if t_list[i] == 1:
            count_potion[x_list[i]] += 1
        else:
            count_monster[x_list[i]] += 1
    for x in range(1, N+1):
        if count_monster[x] > count_potion[x]:
            # can't possibly defeat all
            print(-1)
            return
    
    # Binary Search over K from 0..(number_of_monsters).
    # The maximum number_of_monsters is at most N. We'll gather total_monsters to help bound.
    total_monsters = sum(1 for i in range(1,N+1) if t_list[i]==2)
    low, high = 0, total_monsters
    answer_pick = None
    INF = total_monsters+1
    res = INF
    
    while low <= high:
        mid = (low+high)//2
        pick_res = can_defeat_with_K(mid)
        if pick_res is not None:
            res = mid
            answer_pick = pick_res
            high = mid - 1
        else:
            low = mid + 1
    
    if res == INF:
        # No solution
        print(-1)
        return
    
    # Otherwise, res is the minimal K
    print(res)
    # We need to print the pick decisions for each t=1 event in ascending order of those events
    print(" ".join(map(str, answer_pick)))    


def main():
    solve()

if __name__ == "__main__":
    main()