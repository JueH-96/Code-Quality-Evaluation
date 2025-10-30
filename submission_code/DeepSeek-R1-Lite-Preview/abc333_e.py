def main():
    import sys
    from collections import defaultdict

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    events = []
    idx = 1
    for _ in range(N):
        t = int(data[idx])
        x = int(data[idx+1])
        events.append((t, x))
        idx += 2

    # Collect potion events in order
    potion_events = []
    for i in range(N):
        if events[i][0] == 1:
            potion_events.append(i)

    # Process events in reverse to decide pick up
    needed_potions = defaultdict(int)
    pick_up = [0] * N  # 0-indexed, pick_up[i] = 1 if pick up, else 0

    for i in range(N-1, -1, -1):
        t, x = events[i]
        if t == 2:
            needed_potions[x] += 1
        elif t == 1:
            if needed_potions[x] > 0:
                pick_up[i] = 1
                needed_potions[x] -= 1

    # Simulate the adventure to check if all monsters can be defeated and find K
    held_potions = defaultdict(int)
    current_total = 0
    K = 0

    for i in range(N):
        t, x = events[i]
        if t == 1:
            if pick_up[i]:
                held_potions[x] += 1
                current_total += 1
                if current_total > K:
                    K = current_total
        elif t == 2:
            if held_potions[x] > 0:
                held_potions[x] -= 1
                current_total -= 1
            else:
                print(-1)
                return

    # Collect choices for potion events in order
    choices = [str(pick_up[i]) for i in potion_events]

    print(K)
    print(' '.join(choices))

if __name__ == "__main__":
    main()