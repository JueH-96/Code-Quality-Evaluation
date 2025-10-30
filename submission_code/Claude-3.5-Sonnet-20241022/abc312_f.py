N, M = map(int, input().split())
items = []
for _ in range(N):
    t, x = map(int, input().split())
    items.append((t, x))

def solve(N, M, items):
    # Sort items by happiness in descending order within each type
    pull_tabs = sorted([(x, i) for i, (t, x) in enumerate(items) if t == 0], reverse=True)
    reg_cans = sorted([(x, i) for i, (t, x) in enumerate(items) if t == 1], reverse=True)
    openers = sorted([(x, i) for i, (t, x) in enumerate(items) if t == 2], reverse=True)
    
    max_happiness = 0
    
    # Try each possible number of openers
    for num_openers in range(len(openers) + 1):
        if num_openers > M:
            break
            
        # Calculate total capacity of can openers
        opener_capacity = sum(x for x, _ in openers[:num_openers])
        
        # Try each possible number of regular cans
        for num_reg_cans in range(min(opener_capacity + 1, len(reg_cans) + 1)):
            remaining = M - num_openers - num_reg_cans
            if remaining < 0:
                continue
                
            # Add remaining slots with pull-tab cans
            num_pull_tabs = min(remaining, len(pull_tabs))
            if num_pull_tabs < remaining:
                continue
                
            # Calculate happiness
            happiness = 0
            # Add happiness from regular cans (limited by opener capacity)
            happiness += sum(x for x, _ in reg_cans[:num_reg_cans])
            # Add happiness from pull-tab cans
            happiness += sum(x for x, _ in pull_tabs[:num_pull_tabs])
            
            max_happiness = max(max_happiness, happiness)
            
    return max_happiness

print(solve(N, M, items))