def solve():
    n, m = map(int, input().split())
    items = []
    for _ in range(n):
        items.append(list(map(int, input().split())))

    pull_tab_cans = []
    regular_cans = []
    can_openers = []

    for t, x in items:
        if t == 0:
            pull_tab_cans.append(x)
        elif t == 1:
            regular_cans.append(x)
        else:
            can_openers.append(x)

    pull_tab_cans.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)

    ans = 0
    for i in range(1 << n):
        cur_items = []
        count = 0
        for j in range(n):
            if (i >> j) & 1:
                cur_items.append(items[j])
                count += 1

        if count != m:
            continue

        cur_pull_tab = []
        cur_regular = []
        cur_opener = []
        for t, x in cur_items:
            if t == 0:
                cur_pull_tab.append(x)
            elif t == 1:
                cur_regular.append(x)
            else:
                cur_opener.append(x)

        cur_pull_tab.sort(reverse=True)
        cur_regular.sort(reverse=True)
        cur_opener.sort(reverse=True)

        cur_ans = sum(cur_pull_tab)
        
        opener_capacity = sum(cur_opener)
        
        cur_ans += sum(cur_regular[:min(len(cur_regular), opener_capacity)])

        ans = max(ans, cur_ans)

    print(ans)

solve()