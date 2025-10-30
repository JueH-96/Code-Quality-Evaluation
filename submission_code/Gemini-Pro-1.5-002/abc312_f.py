# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    items = []
    for _ in range(n):
        items.append(list(map(int, input().split())))

    ans = 0
    for i in range(1 << n):
        count = 0
        happiness = 0
        current_items = []
        for j in range(n):
            if (i >> j) & 1:
                count += 1
                current_items.append(items[j])

        if count != m:
            continue

        pull_tabs = []
        regular_cans = []
        can_openers = []

        for item in current_items:
            if item[0] == 0:
                pull_tabs.append(item[1])
            elif item[0] == 1:
                regular_cans.append(item[1])
            else:
                can_openers.append(item[1])
        
        current_happiness = sum(pull_tabs)
        
        regular_cans.sort(reverse=True)
        
        opener_capacity = sum(can_openers)
        
        opened_cans = 0
        for can_happiness in regular_cans:
            if opened_cans < opener_capacity:
                current_happiness += can_happiness
                opened_cans += 1
        
        ans = max(ans, current_happiness)

    print(ans)

solve()