def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    items = []
    index = 2
    for _ in range(N):
        T = int(data[index])
        X = int(data[index+1])
        items.append((T, X))
        index += 2
    
    # Separate items into three categories
    pull_tab = []
    regular = []
    openers = []
    
    for item in items:
        if item[0] == 0:
            pull_tab.append(item[1])
        elif item[0] == 1:
            regular.append(item[1])
        else:
            openers.append(item[1])
    
    # Sort all lists in descending order
    pull_tab.sort(reverse=True)
    regular.sort(reverse=True)
    openers.sort(reverse=True)
    
    # Precompute the prefix sums for pull_tab and regular
    pull_tab_prefix = [0]
    for x in pull_tab:
        pull_tab_prefix.append(pull_tab_prefix[-1] + x)
    
    regular_prefix = [0]
    for x in regular:
        regular_prefix.append(regular_prefix[-1] + x)
    
    # Precompute the total number of openers and their capacities
    total_openers = 0
    opener_capacity = 0
    for x in openers:
        total_openers += 1
        opener_capacity += x
    
    # Determine the maximum number of regular cans we can use
    # We can use min(opener_capacity, number of regular cans, M - number of pull_tab cans)
    # So we need to iterate over the number of pull_tab cans and compute the best combination
    
    max_happiness = 0
    
    # Iterate over the number of pull_tab cans (p)
    for p in range(min(len(pull_tab), M) + 1):
        # Remaining items to choose: M - p
        # We need to choose some regular cans and some openers
        # The number of regular cans we can use is limited by the opener capacity and the number of regular cans
        # Let's denote the number of regular cans as r
        # The number of openers is o = (M - p - r)
        # The total opener capacity must be >= r
        # So r <= min(opener_capacity, len(regular), M - p)
        
        # Calculate the maximum possible r
        max_r = min(opener_capacity, len(regular), M - p)
        
        # Calculate the happiness for this p and r
        # Happiness is pull_tab_prefix[p] + regular_prefix[r]
        # We need to choose the best r for this p
        
        # To find the best r, we can iterate r from 0 to max_r
        # But to optimize, we can precompute the sum of the top r regular cans and the sum of the top (M - p - r) openers
        # However, since openers do not contribute to happiness directly, we only need to ensure that the sum of their capacities is >= r
        
        # So for each p, we can find the maximum r such that the sum of the top (M - p - r) openers' capacities is >= r
        
        # To find the maximum r, we can perform a binary search on r
        low = 0
        high = max_r
        best_r = 0
        
        while low <= high:
            mid = (low + high) // 2
            # Calculate the number of openers needed: o = M - p - mid
            o = M - p - mid
            if o < 0:
                high = mid - 1
                continue
            # Calculate the sum of the top o openers' capacities
            if o > len(openers):
                sum_opener_capacity = sum(openers)
            else:
                sum_opener_capacity = sum(openers[:o])
            if sum_opener_capacity >= mid:
                best_r = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Calculate the happiness for this p and best_r
        happiness = pull_tab_prefix[p] + regular_prefix[best_r]
        if happiness > max_happiness:
            max_happiness = happiness
    
    print(max_happiness)

if __name__ == "__main__":
    main()