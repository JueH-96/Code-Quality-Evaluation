# YOUR CODE HERE
import heapq
import sys
input = sys.stdin.read

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
    
    pull_tab_cans.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)
    
    max_happiness = 0
    current_happiness = 0
    can_openers_used = 0
    can_openers_heap = []
    
    for i in range(min(M, len(pull_tab_cans))):
        current_happiness += pull_tab_cans[i]
        max_happiness = max(max_happiness, current_happiness)
    
    for opener in can_openers:
        heapq.heappush(can_openers_heap, -opener)
    
    for i in range(min(M, len(regular_cans))):
        if can_openers_heap:
            can_openers_used += 1
            opener_capacity = -heapq.heappop(can_openers_heap)
            if opener_capacity > 1:
                heapq.heappush(can_openers_heap, -(opener_capacity - 1))
            current_happiness += regular_cans[i]
            max_happiness = max(max_happiness, current_happiness)
    
    return max_happiness

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    items = [(int(data[i*2+2]), int(data[i*2+3])) for i in range(N)]
    print(max_happiness(N, M, items))

if __name__ == "__main__":
    main()