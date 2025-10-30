def solve():
    n, m = map(int, input().split())
    items = []
    for _ in range(n):
        items.append(list(map(int, input().split())))
    
    pull_tab_cans = []
    regular_cans = []
    can_openers = []
    
    for item in items:
        if item[0] == 0:
            pull_tab_cans.append(item[1])
        elif item[0] == 1:
            regular_cans.append(item[1])
        elif item[0] == 2:
            can_openers.append(item[1])
            
    pull_tab_cans.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)
    
    n_pull_tab = len(pull_tab_cans)
    n_regular = len(regular_cans)
    n_openers = len(can_openers)
    
    max_total_happiness = 0
    
    pull_tab_prefix_sum = [0] * (n_pull_tab + 1)
    for i in range(n_pull_tab):
        pull_tab_prefix_sum[i+1] = pull_tab_prefix_sum[i] + pull_tab_cans[i]
        
    regular_prefix_sum = [0] * (n_regular + 1)
    for i in range(n_regular):
        regular_prefix_sum[i+1] = regular_prefix_sum[i] + regular_cans[i]
        
    can_opener_prefix_sum = [0] * (n_openers + 1)
    for i in range(n_openers):
        can_opener_prefix_sum[i+1] = can_opener_prefix_sum[i] + can_openers[i]
        
    for k in range(min(n_openers, m) + 1):
        can_opener_capacity = can_opener_prefix_sum[k]
        for m0 in range(max(0, m - k - n_regular), min(m - k, n_pull_tab) + 1):
            m1 = m - k - m0
            if 0 <= m1 <= n_regular:
                current_happiness = pull_tab_prefix_sum[m0] + regular_prefix_sum[min(m1, can_opener_capacity)]
                max_total_happiness = max(max_total_happiness, current_happiness)
                
    print(max_total_happiness)

if __name__ == '__main__':
    solve()