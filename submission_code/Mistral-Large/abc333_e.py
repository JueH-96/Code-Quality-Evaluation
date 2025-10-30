import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    events = []

    for i in range(N):
        t = int(data[2 * i + 1])
        x = int(data[2 * i + 2])
        events.append((t, x))

    potion_count = defaultdict(int)
    max_potions = 0
    current_potions = 0
    actions = []

    for t, x in events:
        if t == 1:
            potion_count[x] += 1
            current_potions += 1
            max_potions = max(max_potions, current_potions)
            actions.append(1)
        elif t == 2:
            if potion_count[x] > 0:
                potion_count[x] -= 1
                current_potions -= 1
            else:
                print(-1)
                return

    if any(potion_count[x] < 0 for x in potion_count):
        print(-1)
        return

    print(max_potions)
    print(' '.join(map(str, actions)))

if __name__ == "__main__":
    main()