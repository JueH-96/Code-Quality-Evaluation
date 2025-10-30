import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1

    events = []
    for _ in range(N):
        t = int(input[ptr])
        x = int(input[ptr + 1])
        ptr += 2
        events.append((t, x, _))  # store index as well

    # Separate monsters and potions, and group by x
    monsters = defaultdict(list)
    potions = defaultdict(list)
    for t, x, idx in events:
        if t == 2:
            monsters[x].append(idx)
        else:
            potions[x].append(idx)

    # Check feasibility and collect selected potion indices
    selected = defaultdict(set)
    possible = True
    for x in monsters:
        m_x = len(monsters[x])
        p_x = potions[x]
        if len(p_x) < m_x:
            possible = False
            break
        # Process in reverse to select latest possible
        j = len(p_x) - 1
        selected_potions = []
        for i in reversed(range(m_x)):
            e_i = monsters[x][i]
            while j >= 0 and p_x[j] >= e_i:
                j -= 1
            if j < 0:
                possible = False
                break
            selected_potions.append(p_x[j])
            j -= 1
        if not possible:
            break
        # Sort selected potions in event order
        selected_potions.sort()
        selected[x] = set(selected_potions)
    if not possible:
        print(-1)
        return

    # Simulate to find K_min and check feasibility
    current = defaultdict(int)
    max_count = 0
    for t, x, idx in events:
        if t == 1:
            if idx in selected.get(x, set()):
                current[x] += 1
                if current[x] > max_count:
                    max_count = current[x]
        else:
            if current[x] == 0:
                print(-1)
                return
            current[x] -= 1

    # Collect the pick actions
    picks = []
    for t, x, idx in events:
        if t == 1:
            if idx in selected.get(x, set()):
                picks.append(1)
            else:
                picks.append(0)

    print(max_count)
    print(' '.join(map(str, picks)))

if __name__ == "__main__":
    main()