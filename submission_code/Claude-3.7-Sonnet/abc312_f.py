def max_happiness(N, M, items):
    # Separate items by type
    pull_tab_cans = []
    regular_cans = []
    can_openers = []
    
    for t, x in items:
        if t == 0:
            pull_tab_cans.append(x)  # Pull-tab can: direct happiness
        elif t == 1:
            regular_cans.append(x)   # Regular can: needs opener for happiness
        else:  # t == 2
            can_openers.append(x)    # Can opener: can open X_i cans
    
    # Sort items by value (happiness or capacity)
    pull_tab_cans.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)
    
    max_total_happiness = 0
    
    # Try different combinations of items
    # Iterate through number of can openers to pick
    for c in range(min(M + 1, len(can_openers) + 1)):
        total_capacity = sum(can_openers[:c])  # Total cans we can open
        
        # Iterate through number of regular cans to pick
        for b in range(min(M - c + 1, len(regular_cans) + 1)):
            if b > total_capacity:
                continue  # Skip if we can't open all regular cans
            
            # Calculate happiness from regular cans
            happiness_from_regular_cans = sum(regular_cans[:b])
            
            # Calculate happiness from pull-tab cans
            remaining_picks = M - c - b
            happiness_from_pull_tabs = sum(pull_tab_cans[:min(remaining_picks, len(pull_tab_cans))])
            
            total_happiness = happiness_from_regular_cans + happiness_from_pull_tabs
            max_total_happiness = max(max_total_happiness, total_happiness)
    
    return max_total_happiness

def main():
    N, M = map(int, input().split())
    items = []
    for _ in range(N):
        T, X = map(int, input().split())
        items.append((T, X))
    
    result = max_happiness(N, M, items)
    print(result)

if __name__ == "__main__":
    main()