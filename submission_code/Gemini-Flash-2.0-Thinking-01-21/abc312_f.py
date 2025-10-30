def solve():
    n, m = map(int, input().split())
    items = []
    for _ in range(n):
        items.append(list(map(int, input().split())))
    
    pull_tab_cans = []
    regular_cans = []
    can_openers = []
    
    for item in items:
        item_type, value = item
        if item_type == 0:
            pull_tab_cans.append(value)
        elif item_type == 1:
            regular_cans.append(value)
        elif item_type == 2:
            can_openers.append(value)
            
    pull_tab_cans.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)
    
    max_happiness = 0
    
    for n_openers in range(min(m, len(can_openers)) + 1):
        current_capacity = sum(can_openers[:n_openers])
        remaining_items_to_select = m - n_openers
        if remaining_items_to_select < 0:
            continue
            
        n_pull_tab_cans_selected = min(remaining_items_to_select, len(pull_tab_cans))
        remaining_items_to_select -= n_pull_tab_cans_selected
        if remaining_items_to_select < 0:
            remaining_items_to_select = 0
            
        n_regular_cans_selected = min(remaining_items_to_select, len(regular_cans))
        
        current_happiness = 0
        if n_pull_tab_cans_selected > 0:
            current_happiness += sum(pull_tab_cans[:n_pull_tab_cans_selected])
        
        n_regular_cans_opened = min(n_regular_cans_selected, current_capacity)
        if n_regular_cans_opened > 0:
            current_happiness += sum(regular_cans[:n_regular_cans_opened])
            
        max_happiness = max(max_happiness, current_happiness)
        
    print(max_happiness)

if __name__ == '__main__':
    solve()