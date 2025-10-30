import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    events = []
    for _ in range(N):
        t, x = map(int, sys.stdin.readline().split())
        events.append((t, x))
    
    potion_indices = defaultdict(list)
    monster_indices = defaultdict(list)
    
    for i in range(N):
        t, x = events[i]
        if t == 1:
            potion_indices[x].append(i)
        else:
            monster_indices[x].append(i)
    
    used_potions = set()
    
    for x in potion_indices:
        ptr = len(potion_indices[x]) - 1
        for m_i in reversed(monster_indices[x]):
            found = False
            while ptr >= 0:
                p_i = potion_indices[x][ptr]
                if p_i < m_i:
                    used_potions.add(p_i)
                    ptr -= 1
                    found = True
                    break
                else:
                    ptr -= 1
            if not found:
                print(-1)
                return
    
    inventory = defaultdict(int)
    current_inventory = 0
    max_inventory = 0
    take = [0] * N
    
    for i in range(N):
        t, x = events[i]
        if t == 1:
            if i in used_potions:
                take[i] = 1
                inventory[x] += 1
                current_inventory += 1
            else:
                take[i] = 0
        else:
            if inventory[x] == 0:
                print(-1)
                return
            inventory[x] -= 1
            current_inventory -= 1
        if current_inventory > max_inventory:
            max_inventory = current_inventory
    
    potion_events_order = [i for i in range(N) if events[i][0] == 1]
    output_take = [str(take[i]) for i in potion_events_order]
    
    print(max_inventory)
    print(' '.join(output_take))

if __name__ == "__main__":
    main()