def can_defeat_all(K, events):
    inventory = {}
    current_potions = 0
    for t, x in events:
        if t == 1:
            if current_potions < K:
                inventory[x] = inventory.get(x, 0) + 1
                current_potions += 1
            else:
                continue
        elif t == 2:
            if x in inventory and inventory[x] > 0:
                inventory[x] -= 1
                current_potions -= 1
            else:
                return False
    return True

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    events = []
    idx = 1
    for _ in range(N):
        t = int(data[idx])
        x = int(data[idx + 1])
        events.append((t, x))
        idx += 2
    
    # Binary search for K_min
    left = 0
    right = N
    possible = False
    K_min = -1
    while left <= right:
        mid = (left + right) // 2
        if can_defeat_all(mid, events):
            K_min = mid
            right = mid - 1
            possible = True
        else:
            left = mid + 1
    if not possible:
        print(-1)
        return
    
    # Record the sequence of picks for K_min
    inventory = {}
    current_potions = 0
    picks = [0] * N  # Initialize picks for all events t_i == 1
    pick_idx = 0
    for idx, (t, x) in enumerate(events):
        if t == 1:
            if current_potions < K_min:
                inventory[x] = inventory.get(x, 0) + 1
                current_potions += 1
                picks[pick_idx] = 1
            pick_idx += 1
        elif t == 2:
            if x in inventory and inventory[x] > 0:
                inventory[x] -= 1
                current_potions -= 1
            else:
                # This should not happen since we already checked with can_defeat_all
                assert False
    # Output the results
    print(K_min)
    # Filter picks for t_i == 1 in ascending order
    pick_sequence = [str(picks[i]) for i in range(N) if events[i][0] == 1]
    print(' '.join(pick_sequence))

if __name__ == '__main__':
    main()