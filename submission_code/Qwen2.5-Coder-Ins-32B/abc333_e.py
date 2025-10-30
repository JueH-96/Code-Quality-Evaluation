# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    import heapq
    input = sys.stdin.read().split()
    N = int(input[0])
    events = []
    index = 1
    potions = [0] * (N + 1)
    actions = [0] * N
    for i in range(N):
        t = int(input[index])
        x = int(input[index + 1])
        events.append((t, x, i))
        index += 2

    # First pass to determine if it's possible to defeat all monsters
    for t, x, i in events:
        if t == 1:
            potions[x] += 1
        else:
            if potions[x] > 0:
                potions[x] -= 1
            else:
                print(-1)
                return

    # Reset potions for the second pass
    potions = [0] * (N + 1)
    max_potions = 0
    min_k = float('inf')
    actions = [0] * N
    potion_heap = []

    for t, x, i in events:
        if t == 1:
            heapq.heappush(potion_heap, x)
            max_potions += 1
            min_k = min(min_k, max_potions)
        else:
            if potion_heap:
                used_potion = heapq.heappop(potion_heap)
                actions[i] = 1
                max_potions -= 1
            else:
                print(-1)
                return

    # Determine the minimum K and the corresponding actions
    potions = [0] * (N + 1)
    max_potions = 0
    actions = [0] * N
    potion_heap = []
    for t, x, i in events:
        if t == 1:
            if max_potions < min_k:
                heapq.heappush(potion_heap, x)
                max_potions += 1
                actions[i] = 1
        else:
            if potion_heap:
                used_potion = heapq.heappop(potion_heap)
                actions[i] = 1
                max_potions -= 1
            else:
                print(-1)
                return

    print(min_k)
    print(' '.join(str(actions[i]) for t, x, i in events if t == 1))

if __name__ == "__main__":
    main()