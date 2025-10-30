def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    pull_tab_cans = []
    regular_cans = []
    can_openers = []
    
    index = 2
    for _ in range(N):
        T = int(data[index])
        X = int(data[index + 1])
        if T == 0:
            pull_tab_cans.append(X)
        elif T == 1:
            regular_cans.append(X)
        elif T == 2:
            can_openers.append(X)
        index += 2
    
    # Sort the lists to maximize happiness
    pull_tab_cans.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)
    
    # Use a greedy approach to maximize happiness
    happiness = 0
    used_items = 0
    
    # First, take as many pull-tab cans as possible
    for value in pull_tab_cans:
        if used_items < M:
            happiness += value
            used_items += 1
        else:
            break
    
    # Then, consider using can openers on regular cans
    # We need to find the best combination of regular cans and can openers
    # We can use a priority queue or just sort and use the best options
    import heapq
    
    # Use a max heap for regular cans to get the best happiness
    regular_heap = []
    for value in regular_cans:
        heapq.heappush(regular_heap, -value)
    
    # Track how many regular cans we can open with the available openers
    total_opens = 0
    for opens in can_openers:
        total_opens += opens
    
    # We can only use up to M-used_items regular cans
    max_regular_cans_use = min(M - used_items, total_opens, len(regular_heap))
    
    # Get the best regular cans to use
    regular_happiness = 0
    for _ in range(max_regular_cans_use):
        if regular_heap:
            regular_happiness -= heapq.heappop(regular_heap)
    
    # Calculate the maximum happiness possible
    max_happiness = happiness + regular_happiness
    
    print(max_happiness)

if __name__ == "__main__":
    main()