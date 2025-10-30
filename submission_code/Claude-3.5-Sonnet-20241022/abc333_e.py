def solve():
    N = int(input())
    events = []
    for _ in range(N):
        t, x = map(int, input().split())
        events.append((t, x))
    
    # Try each possible K value
    for max_k in range(N + 1):
        # Try all possible combinations of picking up potions
        def try_combination(event_idx, potions, max_count):
            if max_count > max_k:
                return None
            
            if event_idx == N:
                return []
            
            t, x = events[event_idx]
            
            if t == 1:
                # Try picking up
                potions[x] = potions.get(x, 0) + 1
                pick_result = try_combination(event_idx + 1, potions.copy(), max(max_count, sum(potions.values())))
                if pick_result is not None:
                    return [1] + pick_result
                
                # Try not picking up
                potions[x] = potions.get(x, 0) - 1
                if potions[x] == 0:
                    del potions[x]
                skip_result = try_combination(event_idx + 1, potions.copy(), max_count)
                if skip_result is not None:
                    return [0] + skip_result
                
                return None
                
            else:  # t == 2
                # Need potion to defeat monster
                if x not in potions or potions[x] == 0:
                    return None
                
                potions[x] -= 1
                if potions[x] == 0:
                    del potions[x]
                    
                return try_combination(event_idx + 1, potions.copy(), max_count)
        
        result = try_combination(0, {}, 0)
        if result is not None:
            print(max_k)
            print(*[x for i, (t, _) in enumerate(events) if t == 1])
            return
    
    print(-1)

solve()