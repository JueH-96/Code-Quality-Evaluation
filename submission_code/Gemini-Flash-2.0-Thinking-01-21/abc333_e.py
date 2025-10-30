import collections

def solve():
    n = int(input())
    events = []
    for _ in range(n):
        t, x = map(int, input().split())
        events.append({'type': t, 'potion_type': x})
    
    potion_event_indices = [i for i, event in enumerate(events) if event['type'] == 1]
    num_potion_events = len(potion_event_indices)
    
    min_max_potions = float('inf')
    best_actions = None
    possible = False
    
    import itertools
    
    potion_decisions_options = list(itertools.product([0, 1], repeat=num_potion_events))
    
    valid_actions_list = []
    min_max_k = float('inf')
    best_action_sequence = None
    
    for decisions in potion_decisions_options:
        potion_counts = collections.defaultdict(int)
        max_potions_count = 0
        actions = []
        potion_decision_index = 0
        defeated = False
        current_actions = []
        
        for i in range(n):
            event = events[i]
            if event['type'] == 1:
                potion_type = event['potion_type']
                decision = decisions[potion_decision_index]
                potion_decision_index += 1
                if decision == 1:
                    potion_counts[potion_type] += 1
                    current_actions.append(1)
                else:
                    current_actions.append(0)
            elif event['type'] == 2:
                monster_type = event['potion_type']
                if potion_counts[monster_type] > 0:
                    potion_counts[monster_type] -= 1
                else:
                    defeated = True
                    break
            max_potions_count = max(max_potions_count, sum(potion_counts.values()))
            
        if not defeated:
            possible = True
            current_max_k = max_potions_count
            if current_max_k < min_max_k:
                min_max_k = current_max_k
                action_sequence_for_output = []
                potion_action_index = 0
                for event in events:
                    if event['type'] == 1:
                        action_sequence_for_output.append(current_actions[potion_action_index])
                        potion_action_index += 1
                best_action_sequence = action_sequence_for_output
                
    if not possible:
        print("-1")
    else:
        print(min_max_k)
        print(*(best_action_sequence))

if __name__ == '__main__':
    solve()