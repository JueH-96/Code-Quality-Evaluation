def solve():
    n, m = map(int, input().split())
    items = []
    for _ in range(n):
        items.append(list(map(int, input().split())))

    max_happiness = 0

    for i in range(1 << n):
        if bin(i).count('1') != m:
            continue

        current_happiness = 0
        selected_items = []
        for j in range(n):
            if (i >> j) & 1:
                selected_items.append(items[j])

        pull_tab_cans = []
        regular_cans = []
        can_openers = []

        for item in selected_items:
            if item[0] == 0:
                pull_tab_cans.append(item[1])
            elif item[0] == 1:
                regular_cans.append(item[1])
            else:
                can_openers.append(item[1])
        
        current_happiness = sum(pull_tab_cans)
        
        regular_cans.sort(reverse=True)
        can_openers.sort(reverse=True)
        
        opener_idx = 0
        can_idx = 0
        
        while can_idx < len(regular_cans) and opener_idx < len(can_openers):
            
            if can_openers[opener_idx] > 0:
                current_happiness += regular_cans[can_idx]
                can_openers[opener_idx] -= 1
                can_idx += 1
            else:
                opener_idx += 1
        
        max_happiness = max(max_happiness, current_happiness)

    print(max_happiness)

solve()