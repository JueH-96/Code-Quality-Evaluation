import collections

def solve():
    n = int(input())
    events = []
    for _ in range(n):
        line = list(map(int, input().split()))
        events.append({'type': line[0], 'potion_type': line[1]})
    
    potion_event_indices = []
    monster_counts = collections.defaultdict(int)
    potion_counts_by_type = collections.defaultdict(int)
    
    for i in range(n):
        event = events[i]
        if event['type'] == 1:
            potion_event_indices.append(i)
            potion_counts_by_type[event['potion_type']] += 1
        elif event['type'] == 2:
            monster_counts[event['potion_type']] += 1
            
    for potion_type in monster_counts:
        if potion_counts_by_type[potion_type] < monster_counts[potion_type]:
            print("-1")
            return
            
    min_max_potions = float('inf')
    best_actions = None
    
    num_potion_events = len(potion_event_indices)
    
    for i in range(1 << num_potion_events):
        current_actions = []
        potion_inventory = collections.defaultdict(int)
        max_potions_count = 0
        possible = True
        action_index = 0
        potion_actions_for_output = []
        
        for event_index in range(n):
            event = events[event_index]
            if event['type'] == 1:
                potion_type = event['potion_type']
                decision = (i >> action_index) & 1
                action_index += 1
                if decision == 1:
                    potion_inventory[potion_type] += 1
                    potion_actions_for_output.append(1)
                else:
                    potion_actions_for_output.append(0)
            elif event['type'] == 2:
                monster_type = event['potion_type']
                if potion_inventory[monster_type] > 0:
                    potion_inventory[monster_type] -= 1
                else:
                    possible = False
                    break
            current_potions_sum = sum(potion_inventory.values())
            max_potions_count = max(max_potions_count, current_potions_sum)
            
        if possible:
            if max_potions_count < min_max_potions:
                min_max_potions = max_potions_count
                best_actions = potion_actions_for_output
                
    if best_actions is None:
        print("-1")
    else:
        print(min_max_potions)
        print(*(best_actions))

if __name__ == '__main__':
    solve()