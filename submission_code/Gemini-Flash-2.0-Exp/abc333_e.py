def solve():
    n = int(input())
    events = []
    for _ in range(n):
        events.append(list(map(int, input().split())))

    def check(actions):
        inventory = {}
        for i in range(1, n + 1):
            inventory[i] = 0
        
        k = 0
        max_k = 0
        
        action_idx = 0
        
        for i in range(n):
            t, x = events[i]
            
            if t == 1:
                if actions[action_idx] == 1:
                    inventory[x] += 1
                    k += 1
                    max_k = max(max_k, k)
                action_idx += 1
            else:
                if inventory[x] > 0:
                    inventory[x] -= 1
                    k -= 1
                else:
                    return False, -1
        
        return True, max_k
    
    potion_indices = [i for i in range(n) if events[i][0] == 1]
    num_potions = len(potion_indices)
    
    min_k = float('inf')
    best_actions = None
    
    for i in range(2**num_potions):
        actions = []
        for j in range(num_potions):
            if (i >> j) & 1:
                actions.append(1)
            else:
                actions.append(0)
        
        success, max_k = check(actions)
        
        if success:
            if max_k < min_k:
                min_k = max_k
                best_actions = actions
    
    if min_k == float('inf'):
        print("-1")
    else:
        print(min_k)
        
        action_idx = 0
        result = []
        for i in range(n):
            if events[i][0] == 1:
                result.append(str(best_actions[action_idx]))
                action_idx += 1
        print(" ".join(result))

solve()