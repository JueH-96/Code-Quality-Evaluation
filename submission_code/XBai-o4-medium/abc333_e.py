import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    events = []  # list of (t_i, x_i) for each event
    for _ in range(N):
        t = int(input[ptr])
        x = int(input[ptr+1])
        events.append( (t, x) )
        ptr += 2

    # Build events_by_x: for each x, list of (type, original_index)
    events_by_x = defaultdict(list)
    for idx in range(N):
        t, x = events[idx]
        events_by_x[x].append( (t, idx) )

    ans = [0] * N  # for each event, 0 or 1 (only for type 1)

    for x in events_by_x:
        event_list = events_by_x[x]
        M = len(event_list)
        decisions = [0] * M
        prev_r = 0
        current_monsters_count = 0
        collects_since_last_r = []
        for i in range(M):
            t, original_idx = event_list[i]
            if t == 1:
                collects_since_last_r.append(i)
            else:  # t == 2
                current_monsters_count += 1
                new_r = max(prev_r, current_monsters_count)
                if new_r > prev_r:
                    need_to_take = new_r - prev_r
                    if len(collects_since_last_r) < need_to_take:
                        print(-1)
                        return
                    take_from = len(collects_since_last_r) - need_to_take
                    for j in range(take_from, len(collects_since_last_r)):
                        collect_idx_in_event_list = collects_since_last_r[j]
                        decisions[collect_idx_in_event_list] = 1
                    prev_r = new_r
                    collects_since_last_r = []
        # Fill ans for this x's events
        for i in range(M):
            t, original_idx = event_list[i]
            if t == 1:
                ans[original_idx] = decisions[i]

    # Now simulate to compute max_k and check validity
    current_potions = defaultdict(int)
    max_k = 0
    current_total = 0

    for i in range(N):
        t, x = events[i]
        if t == 1:
            if ans[i] == 1:
                current_potions[x] += 1
                current_total += 1
                if current_total > max_k:
                    max_k = current_total
        else:  # t == 2
            if current_potions[x] == 0:
                # This should not happen if the previous processing was correct
                print(-1)
                return
            current_potions[x] -= 1
            current_total -= 1

    # Now collect the output for type 1 events in order
    output_actions = []
    for i in range(N):
        if events[i][0] == 1:
            output_actions.append( str(ans[i]) )

    print(max_k)
    print(' '.join(output_actions))

if __name__ == '__main__':
    main()