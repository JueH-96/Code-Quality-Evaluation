# YOUR CODE HERE
import sys
import heapq

def max_happiness(N, M, items):
    pull_tab_cans = []
    regular_cans = []
    can_openers = []
    
    for T, X in items:
        if T == 0:
            pull_tab_cans.append(X)
        elif T == 1:
            regular_cans.append(X)
        elif T == 2:
            can_openers.append(X)
    
    # Sort pull-tab cans and regular cans in descending order of happiness
    pull_tab_cans.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)
    
    # Take the top M pull-tab cans if possible
    max_happiness = sum(pull_tab_cans[:M])
    remaining_items = M - len(pull_tab_cans[:M])
    
    # Use can openers to open regular cans
    if remaining_items > 0:
        # Priority queue (max-heap) for available regular cans
        regular_cans_heap = []
        regular_cans_index = 0
        
        # Iterate over can openers
        for opener_capacity in can_openers:
            # Push available regular cans into the heap
            while regular_cans_index < len(regular_cans):
                heapq.heappush(regular_cans_heap, -regular_cans[regular_cans_index])
                regular_cans_index += 1
            
            # Use the can opener to open the best available regular cans
            for _ in range(opener_capacity):
                if regular_cans_heap and remaining_items > 0:
                    max_happiness -= heapq.heappop(regular_cans_heap)
                    remaining_items -= 1
    
    return max_happiness

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    items = []
    
    index = 2
    for _ in range(N):
        T = int(data[index])
        X = int(data[index + 1])
        items.append((T, X))
        index += 2
    
    result = max_happiness(N, M, items)
    print(result)