import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input())
    events = []
    for _ in range(N):
        t,x = map(int, input().split())
        events.append((t,x))

    # For each type, maintain a stack of pickup event positions.
    stacks = {}
    # Mark which pickup events are used to defeat a monster.
    used = [0]*N

    for i, (t,x) in enumerate(events):
        if t == 1:
            # push this pickup onto the stack of its type
            if x not in stacks:
                stacks[x] = []
            stacks[x].append(i)
        else:
            # monster of type x: need a prior unused pickup
            if x not in stacks or not stacks[x]:
                print(-1)
                return
            # match with the most recent pickup <= i
            pick_pos = stacks[x].pop()
            used[pick_pos] = 1

    # Now we have matched pickups. Any pickup with used=1 is actually picked.
    # Simulate to find peak inventory.
    inventory = 0
    peak = 0
    for i, (t,x) in enumerate(events):
        if t == 1:
            # if we matched it to a future monster, we pick it up
            if used[i]:
                inventory += 1
                if inventory > peak:
                    peak = inventory
        else:
            # encounter monster, use one potion
            inventory -= 1

    # Output
    print(peak)
    # Print actions for each pickup event in order
    # 1 = pick up, 0 = discard
    res = []
    for i, (t,x) in enumerate(events):
        if t == 1:
            res.append('1' if used[i] else '0')
    print(" ".join(res))


if __name__ == "__main__":
    main()