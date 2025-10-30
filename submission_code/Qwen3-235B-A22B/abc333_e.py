import bisect
from collections import defaultdict
import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    events = []
    for _ in range(N):
        t = int(input[idx])
        x = int(input[idx + 1])
        events.append((t, x))
        idx += 2

    type_1_events = defaultdict(list)
    monster_list = []
    for i, (t, x) in enumerate(events):
        if t == 1:
            type_1_events[x].append(i)
        else:
            monster_list.append((x, i))

    taken = [False] * N
    pos = {}
    for x in type_1_events:
        pos[x] = len(type_1_events[x]) - 1

    for x, m_idx in monster_list:
        if x not in type_1_events:
            print(-1)
            return
        ev_list = type_1_events[x]
        current_pos = pos[x]
        if current_pos < 0:
            print(-1)
            return

        i = bisect.bisect_right(ev_list, m_idx) - 1
        if i < 0:
            print(-1)
            return
        if i > current_pos:
            i = current_pos
        if ev_list[i] > m_idx:
            print(-1)
            return

        selected = ev_list[i]
        taken[selected] = True
        pos[x] = i - 1

    current_inventory = 0
    max_inventory = 0
    for i, (t, _) in enumerate(events):
        if t == 1:
            if taken[i]:
                current_inventory += 1
                if current_inventory > max_inventory:
                    max_inventory = current_inventory
        else:
            current_inventory -= 1

    actions = []
    for i, (t, _) in enumerate(events):
        if t == 1:
            actions.append('1' if taken[i] else '0')
    print(max_inventory)
    print(' '.join(actions))

if __name__ == "__main__":
    main()