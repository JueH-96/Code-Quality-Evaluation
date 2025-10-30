import heapq

def main():
    N, M = map(int, input().split())
    pull_tab_cans = []
    regular_cans = []
    can_openers = []
    
    for _ in range(N):
        T, X = map(int, input().split())
        if T == 0:
            pull_tab_cans.append(X)
        elif T == 1:
            regular_cans.append(X)
        else:
            can_openers.append(X)
    
    # Sort the lists in descending order of happiness/utility
    pull_tab_cans.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)
    
    # Use a max heap for can openers to get the one with the highest utility
    can_openers_heap = [(-x, x) for x in can_openers]
    heapq.heapify(can_openers_heap)
    
    happiness = 0
    opened_cans = 0
    for _ in range(M):
        # If there are pull-tab cans, take the one with the highest happiness
        if pull_tab_cans:
            happiness += pull_tab_cans.pop(0)
        # If there are no pull-tab cans but there are regular cans and can openers
        elif regular_cans and can_openers_heap:
            # Check if the current can opener can still be used
            if opened_cans < can_openers_heap[0][1]:
                happiness += regular_cans.pop(0)
                opened_cans += 1
            # If the current can opener is used up, get the next one
            else:
                heapq.heappop(can_openers_heap)
                opened_cans = 1
                happiness += regular_cans.pop(0)
        # If there are no pull-tab cans and no can openers, stop the process
        else:
            break
    
    print(happiness)

if __name__ == "__main__":
    main()