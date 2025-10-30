# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    import heapq

    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    items = []

    for i in range(N):
        T = int(input[2 + 2 * i])
        X = int(input[3 + 2 * i])
        items.append((T, X))

    pull_tab_cans = []
    regular_cans = []
    can_openers = []

    for T, X in items:
        if T == 0:
            pull_tab_cans.append(X)
        elif T == 1:
            regular_cans.append(X)
        else:
            can_openers.append(X)

    pull_tab_cans.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)

    max_happiness = 0
    pull_tab_used = 0
    regular_used = 0
    opener_used = 0

    pull_tab_heap = []
    regular_heap = []

    for i in range(M):
        if pull_tab_used < len(pull_tab_cans):
            heapq.heappush(pull_tab_heap, -pull_tab_cans[pull_tab_used])
            pull_tab_used += 1
        if regular_used < len(regular_cans):
            heapq.heappush(regular_heap, -regular_cans[regular_used])
            regular_used += 1

    while opener_used < len(can_openers) and regular_heap:
        opener_capacity = can_openers[opener_used]
        while opener_capacity > 0 and regular_heap:
            max_happiness -= heapq.heappop(regular_heap)
            opener_capacity -= 1
        opener_used += 1

    while pull_tab_heap:
        max_happiness -= heapq.heappop(pull_tab_heap)

    print(max_happiness)

if __name__ == "__main__":
    main()