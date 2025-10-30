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
    
    # Precompute prefix sums for pull_tab and regular
    pull_tab_prefix = [0]
    for x in pull_tab:
        pull_tab_prefix.append(pull_tab_prefix[-1] + x)
    
    regular_prefix = [0]
    for x in regular:
        regular_prefix.append(regular_prefix[-1] + x)
    
    # Precompute the total number of openers and their capacities
    total_openers = len(openers)
    opener_capacity = sum(openers)
    
    # We need to choose M items, which can be a combination of pull_tab, regular, and openers
    # The goal is to maximize the sum of pull_tab and the sum of regular that can be opened with the available openers
    
    # Let's consider all possible numbers of openers we can take (from 0 to min(total_openers, M))
    max_happiness = 0
    
    for num_openers in range(min(total_openers, M) + 1):
        # Number of items left to choose after selecting num_openers openers
        remaining = M - num_openers
        
        # The number of regular cans we can open is limited by the sum of the capacities of the selected openers
        # We need to select the top num_openers openers
        if num_openers > 0:
            selected_openers = openers[:num_openers]
            total_capacity = sum(selected_openers)
        else:
            total_capacity = 0
        
        # The number of regular cans we can open is min(total_capacity, len(regular))
        # We need to select the top min(total_capacity, len(regular)) regular cans
        num_regular = min(total_capacity, len(regular))
        
        # The remaining items can be pull_tab cans
        num_pull_tab = remaining
        
        # Ensure that num_pull_tab does not exceed the number of pull_tab cans
        num_pull_tab = min(num_pull_tab, len(pull_tab))
        
        # Calculate the total happiness
        happiness = pull_tab_prefix[num_pull_tab] + regular_prefix[num_regular]
        
        if happiness > max_happiness:
            max_happiness = happiness
    
    print(max_happiness)

if __name__ == "__main__":
    main()